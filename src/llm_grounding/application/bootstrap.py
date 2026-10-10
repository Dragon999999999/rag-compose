"""Construct Phase 1 application services from environment-backed infrastructure."""

from sqlalchemy.ext.asyncio import AsyncEngine

from llm_grounding.application.chat_service import ChatService
from llm_grounding.application.document_summary_service import DocumentSummaryService
from llm_grounding.application.project_service import ProjectService
from llm_grounding.assessment.source_reference import SourceReferenceAssessor
from llm_grounding.decision.basic_grounding import BasicGroundingPolicy
from llm_grounding.embedding.sentence_transformers import SentenceTransformersEmbedder
from llm_grounding.infrastructure.config.models import ArchitectureConfig
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
        self.runner = runner
        self.architecture = architecture
        self.ingestion_config = ingestion_config

    async def close(self) -> None:
        """Dispose database and Qdrant client resources after application shutdown."""
        await self.summaries.close()
        await self._vectors.close()
        await self._engine.dispose()


def create_phase_one_application(
    architecture: ArchitectureConfig,
    ingestion_config: IngestionConfig,
    diagnostics: GroundingDiagnostics | None = None,
) -> PhaseOneApplication:
    """Construct PostgreSQL/MinIO/Qdrant Basic RAG services from typed configuration.

    Raises:
        RuntimeError: If configuration selects unsupported Phase 1 components.
    """
    if architecture.name != "basic_rag_v1" or architecture.retriever.type != "vector":
        raise RuntimeError(
            "Phase 1 application requires the basic_rag_v1 vector architecture"
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
    embedder = SentenceTransformersEmbedder(ingestion_config.embedding)
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
    vector = VectorRetriever(embedder, vectors, top_k)
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
    )
    max_history = (
        architecture.chat_history.max_messages if architecture.chat_history else 10
    )
    summaries = DocumentSummaryService(repository, runner)
    return PhaseOneApplication(
        ProjectService(repository, ingestion, vectors, summaries),
        ChatService(repository, runner, max_history),
        engine,
        vectors,
        repository,
        artifacts,
        summaries,
        runner,
        architecture,
        ingestion_config,
    )
