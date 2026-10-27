"""Construct Phase 1 application services from environment-backed infrastructure."""

from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncEngine

from llm_grounding.application.chat_service import ChatService
from llm_grounding.application.document_summary_service import DocumentSummaryService
from llm_grounding.application.eval_corpus_service import EvalCorpusService
from llm_grounding.application.pipeline_runtime import PipelineRuntimeResolver
from llm_grounding.application.pipeline_service import PipelineService
from llm_grounding.application.pipeline_sync import PipelineSynchronizer
from llm_grounding.application.project_service import ProjectService
from llm_grounding.assessment.source_reference import SourceReferenceAssessor
from llm_grounding.decision.basic_grounding import BasicGroundingPolicy
from llm_grounding.domain.corpus import Pipeline
from llm_grounding.embedding.factory import build_embedder
from llm_grounding.infrastructure.config.loader import load_model_catalog
from llm_grounding.infrastructure.config.models import (
    ArchitectureConfig,
    ModelCatalog,
    PipelineConfig,
    VectorBackendConfig,
)
from llm_grounding.infrastructure.config.settings import MinioSettings
from llm_grounding.infrastructure.database.repositories import PostgresRepository
from llm_grounding.infrastructure.database.session import create_session_factory
from llm_grounding.infrastructure.llm.aqueduct import AqueductModelClient
from llm_grounding.infrastructure.prompts import PromptLibrary
from llm_grounding.infrastructure.storage.minio import MinioArtifactStore
from llm_grounding.infrastructure.storage.protocol import ArtifactStore
from llm_grounding.infrastructure.vector.factory import VectorStoreFactory
from llm_grounding.infrastructure.vector.protocol import VectorStore
from llm_grounding.ingestion.models import IngestionConfig
from llm_grounding.ingestion.service import DocumentIngestionService
from llm_grounding.knowledge.chunking.block_aware import BlockAwareChunker
from llm_grounding.knowledge.chunking.fixed_window import FixedWindowChunker
from llm_grounding.orchestration.basic_rag import BasicRagRunner
from llm_grounding.orchestration.grounding_diagnostics import GroundingDiagnostics
from llm_grounding.orchestration.registry import ComponentRegistry
from llm_grounding.parsing.pymupdf import PyMuPDFDocumentParser
from llm_grounding.reasoning.basic_grounded import BasicGroundedReasoner
from llm_grounding.rendering.grounded_answer import GroundedAnswerRenderer
from llm_grounding.retrieval.vector import SummaryRetriever, VectorRetriever


class PhaseOneApplication:
    """Hold Phase 1 services and infrastructure clients that require shutdown."""

    def __init__(
        self,
        projects: ProjectService,
        chats: ChatService,
        engine: AsyncEngine,
        vectors: VectorStore | None,
        repository: PostgresRepository,
        artifacts: ArtifactStore,
        summaries: DocumentSummaryService | None,
        runner: BasicRagRunner | None = None,
        architecture: ArchitectureConfig | None = None,
        ingestion_config: IngestionConfig | None = None,
        pipeline_sync: PipelineSynchronizer | None = None,
        pipeline_service: PipelineService | None = None,
    ) -> None:
        """Expose application use cases while retaining infrastructure ownership."""
        self.projects, self.chats, self._engine, self._vectors = (
            projects,
            chats,
            engine,
            vectors,
        )
        self.repository, self.artifacts, self.summaries = (
            repository,
            artifacts,
            summaries,
        )
        self.evaluation = EvalApplication(EvalCorpusService(repository), vectors)
        self.runner = runner
        self.architecture = architecture
        self.ingestion_config = ingestion_config
        self._pipeline_sync = pipeline_sync
        self.pipeline_service = pipeline_service
        self.pipeline_runtime = pipeline_service.runtime if pipeline_service else None
        self.pipelines: tuple[Pipeline, ...] = ()

    async def startup(self) -> None:
        """Synchronize configured pipelines and expose PostgreSQL-backed records."""
        if self._pipeline_sync is not None:
            await self._pipeline_sync.synchronize()
        assign_default = getattr(
            self.repository, "assign_default_pipeline_to_unassigned_chats", None
        )
        if assign_default is not None:
            await assign_default()
        assign_ingestion = getattr(
            self.repository, "assign_default_ingestion_to_unassigned_projects", None
        )
        if assign_ingestion is not None:
            await assign_ingestion()
        backfill_capabilities = getattr(
            self.repository, "backfill_project_index_capabilities", None
        )
        if backfill_capabilities is not None:
            await backfill_capabilities()
        self.pipelines = await self.repository.list_pipelines()

    async def close(self) -> None:
        """Dispose database and Qdrant client resources after application shutdown."""
        if self.summaries is not None:
            await self.summaries.close()
        if self._vectors is not None and hasattr(self._vectors, "close"):
            await self._vectors.close()
        await self._engine.dispose()


class EvalApplication:
    """Evaluation-only services and derived-index access isolated from projects."""

    def __init__(
        self, corpus_service: EvalCorpusService, embedding_store: VectorStore | None
    ) -> None:
        """Expose corpus lifecycle and optional evaluation vector access."""
        self.corpus_service = corpus_service
        self.embedding_store = embedding_store


def create_phase_one_application(
    architecture: ArchitectureConfig | PipelineConfig,
    ingestion_config: IngestionConfig | None = None,
    diagnostics: GroundingDiagnostics | None = None,
    *,
    model_catalog: ModelCatalog | None = None,
    pipeline_directory: Path | None = None,
) -> PhaseOneApplication:
    """Construct PostgreSQL/MinIO/Qdrant Basic RAG services from typed configuration.

    Raises:
        RuntimeError: If configuration selects unsupported Phase 1 components.
    """
    if isinstance(architecture, PipelineConfig):
        if architecture.kind != "query":
            raise ValueError("Phase One requires a query pipeline")
        catalog = model_catalog or load_model_catalog()
        model_catalog = catalog
        if ingestion_config is None:
            raise ValueError("A query pipeline requires an ingestion configuration")
        architecture, ingestion_config = architecture.resolve_query(
            catalog, ingestion_config
        )
    if ingestion_config is None:
        raise ValueError("An ingestion configuration is required")
    if (
        architecture.reasoner.type != "basic_grounded"
        or architecture.renderer.type != "grounded_answer"
    ):
        raise RuntimeError(
            "Phase 1 application requires a basic_grounded grounded-answer pipeline"
        )
    if architecture.reasoner.prompt_id is None:
        raise RuntimeError("Phase 1 grounded reasoning requires a configured prompt_id")
    minio_settings = MinioSettings.from_environment()
    engine, sessions = create_session_factory()
    repository = PostgresRepository(sessions)
    vectors: VectorStore | None = None
    if ingestion_config.embedding is not None:
        if ingestion_config.vector_backend is None:
            raise RuntimeError("Dense ingestion requires a resolved vector backend")
        vectors = VectorStoreFactory().create(
            VectorBackendConfig(
                type=ingestion_config.vector_backend,
                index_version=ingestion_config.index_version,
            ),
            repository,
        )
    catalog = model_catalog or load_model_catalog()
    pipeline_sync = PipelineSynchronizer(repository, pipeline_directory, catalog)
    pipeline_runtime = PipelineRuntimeResolver(repository, vectors, catalog)
    pipeline_service = PipelineService(repository, catalog, runtime=pipeline_runtime)
    embedder = (
        build_embedder(ingestion_config.embedding)
        if ingestion_config.embedding is not None
        else None
    )
    registry = ComponentRegistry()
    knowledge_builder = registry.build_knowledge_builder(
        ingestion_config.knowledge_builder
    )
    chunker = (
        BlockAwareChunker(ingestion_config.chunker)
        if ingestion_config.chunker.type == "block_aware"
        else FixedWindowChunker(ingestion_config.chunker)
    )
    artifacts = MinioArtifactStore(minio_settings)
    ingestion = DocumentIngestionService(
        repository,
        artifacts,
        PyMuPDFDocumentParser(),
        knowledge_builder,
        chunker,
        embedder,
        vectors,
    )
    runner = None
    summaries = None
    if (
        architecture.retriever.type == "vector"
        and ingestion_config.embedding is not None
        and vectors is not None
        and embedder is not None
    ):
        top_k = (
            architecture.reranker.retrieval_top_k
            or architecture.retriever.top_k
            or 8
        )
        vector = VectorRetriever(
            embedder, vectors, top_k, embedding_config=ingestion_config.embedding
        )
        summary = SummaryRetriever(
            repository,
            vector,
            architecture.retriever.context_chars or 12000,
            architecture.retriever.summary_per_facet or 3,
        )
        reasoner = BasicGroundedReasoner(
            AqueductModelClient(architecture.reasoner.model),
            PromptLibrary.default(),
            architecture.reasoner.prompt_id,
        )
        reranker = registry.build_reranker(architecture.reranker)
        runner = BasicRagRunner(
            vector,
            summary,
            reasoner,
            SourceReferenceAssessor(),
            BasicGroundingPolicy(),
            GroundedAnswerRenderer(),
            diagnostics,
            ingestion_config.embedding,
            reranker=reranker,
            reranker_config=architecture.reranker,
            reranker_builder=registry.build_reranker,
        )
        summaries = DocumentSummaryService(repository, runner)
    max_history = (
        architecture.chat_history.max_messages if architecture.chat_history else 10
    )
    return PhaseOneApplication(
        ProjectService(
            repository,
            ingestion,
            vectors,
            summaries,
            pipeline_service,
        ),
        ChatService(
            repository,
            runner,
            max_history,
            pipeline_runtime,
            pipeline_service,
        ),
        engine,
        vectors,
        repository,
        artifacts,
        summaries,
        runner,
        architecture,
        ingestion_config,
        pipeline_sync,
        pipeline_service,
    )
