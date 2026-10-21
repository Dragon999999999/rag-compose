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
)
from llm_grounding.infrastructure.config.settings import (
    MinioSettings,
    QdrantSettings,
)
from llm_grounding.infrastructure.database.repositories import PostgresRepository
from llm_grounding.infrastructure.database.session import create_session_factory
from llm_grounding.infrastructure.llm.aqueduct import AqueductModelClient
from llm_grounding.infrastructure.prompts import PromptLibrary
from llm_grounding.infrastructure.storage.minio import MinioArtifactStore
from llm_grounding.infrastructure.storage.protocol import ArtifactStore
from llm_grounding.infrastructure.vector.qdrant import QdrantVectorStore
from llm_grounding.ingestion.models import IngestionConfig
from llm_grounding.ingestion.service import DocumentIngestionService
from llm_grounding.knowledge.basic import BasicKnowledgeBuilder
from llm_grounding.knowledge.chunking.block_aware import BlockAwareChunker
from llm_grounding.knowledge.chunking.fixed_window import FixedWindowChunker
from llm_grounding.orchestration.basic_rag import BasicRagRunner
from llm_grounding.orchestration.grounding_diagnostics import GroundingDiagnostics
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
        vectors: QdrantVectorStore,
        repository: PostgresRepository,
        artifacts: ArtifactStore,
        summaries: DocumentSummaryService,
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
        self.pipelines = await self.repository.list_pipelines()

    async def close(self) -> None:
        """Dispose database and Qdrant client resources after application shutdown."""
        await self.summaries.close()
        await self._vectors.close()
        await self._engine.dispose()


class EvalApplication:
    """Evaluation-only services and derived-index access isolated from projects."""

    def __init__(
        self, corpus_service: EvalCorpusService, embedding_store: QdrantVectorStore
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
        catalog = model_catalog or load_model_catalog()
        model_catalog = catalog
        architecture, ingestion_config = architecture.resolve(catalog)
    if ingestion_config is None:
        raise ValueError("An ingestion configuration is required")
    if (
        architecture.retriever.type != "vector"
        or architecture.reasoner.type != "basic_grounded"
        or architecture.renderer.type != "grounded_answer"
    ):
        raise RuntimeError(
            "Phase 1 application requires a vector/basic_grounded pipeline"
        )
    if ingestion_config.vector_backend != "qdrant":
        raise RuntimeError("Phase 1 application currently supports only qdrant")
    if architecture.reasoner.prompt_id is None:
        raise RuntimeError("Phase 1 grounded reasoning requires a configured prompt_id")
    minio_settings = MinioSettings.from_environment()
    qdrant_settings = QdrantSettings.from_environment()
    engine, sessions = create_session_factory()
    repository = PostgresRepository(sessions)
    vectors = QdrantVectorStore(qdrant_settings, repository)
    catalog = model_catalog or load_model_catalog()
    pipeline_sync = PipelineSynchronizer(repository, pipeline_directory, catalog)
    pipeline_runtime = PipelineRuntimeResolver(repository, vectors, catalog)
    pipeline_service = PipelineService(repository, catalog, runtime=pipeline_runtime)
    embedder = build_embedder(ingestion_config.embedding)
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
        BasicKnowledgeBuilder(),
        chunker,
        embedder,
        vectors,
    )
    top_k = architecture.retriever.top_k or 8
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
    runner = BasicRagRunner(
        vector,
        summary,
        reasoner,
        SourceReferenceAssessor(),
        BasicGroundingPolicy(),
        GroundedAnswerRenderer(),
        diagnostics,
        ingestion_config.embedding,
    )
    max_history = (
        architecture.chat_history.max_messages if architecture.chat_history else 10
    )
    summaries = DocumentSummaryService(repository, runner)
    return PhaseOneApplication(
        ProjectService(repository, ingestion, vectors, summaries),
        ChatService(repository, runner, max_history, pipeline_runtime),
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
