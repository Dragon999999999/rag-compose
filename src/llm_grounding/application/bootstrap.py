"""Construct Phase 1 application services from environment-backed infrastructure."""

import os
from pathlib import Path

from llm_grounding.application.chat_service import ChatService
from llm_grounding.application.project_service import ProjectService
from llm_grounding.assessment.source_reference import SourceReferenceAssessor
from llm_grounding.decision.basic_grounding import BasicGroundingPolicy
from llm_grounding.embedding.sentence_transformers import SentenceTransformersEmbedder
from llm_grounding.infrastructure.config.models import ArchitectureConfig
from llm_grounding.infrastructure.database.repositories import PostgresRepository
from llm_grounding.infrastructure.database.session import create_session_factory
from llm_grounding.infrastructure.llm.aqueduct import AqueductModelClient
from llm_grounding.infrastructure.storage.local import LocalArtifactStore
from llm_grounding.infrastructure.vector.pgvector import PgVectorStore
from llm_grounding.ingestion.models import IngestionConfig
from llm_grounding.ingestion.service import DocumentIngestionService
from llm_grounding.knowledge.basic import BasicKnowledgeBuilder
from llm_grounding.knowledge.chunking.block_aware import BlockAwareChunker
from llm_grounding.knowledge.chunking.fixed_window import FixedWindowChunker
from llm_grounding.orchestration.basic_rag import BasicRagRunner
from llm_grounding.parsing.pymupdf import PyMuPDFDocumentParser
from llm_grounding.reasoning.basic_grounded import BasicGroundedReasoner
from llm_grounding.rendering.grounded_answer import GroundedAnswerRenderer
from llm_grounding.retrieval.vector import SummaryRetriever, VectorRetriever


class PhaseOneApplication:
    """Hold thin project/chat services and the database engine that owns their lifecycle."""

    def __init__(
        self, projects: ProjectService, chats: ChatService, engine: object
    ) -> None:
        """Expose application use cases while retaining engine ownership for shutdown."""
        self.projects, self.chats, self._engine = projects, chats, engine

    async def close(self) -> None:
        """Dispose database connections after a CLI invocation or application shutdown."""
        await self._engine.dispose()  # type: ignore[attr-defined]


def create_phase_one_application(
    architecture: ArchitectureConfig, ingestion_config: IngestionConfig
) -> PhaseOneApplication:
    """Construct PostgreSQL/pgvector Basic RAG services from typed configuration.

    Raises:
        RuntimeError: If configuration selects unsupported Phase 1 components.
    """
    if architecture.name != "basic_rag_v1" or architecture.retriever.type != "vector":
        raise RuntimeError(
            "Phase 1 application requires the basic_rag_v1 vector architecture"
        )
    if ingestion_config.vector_backend != "pgvector":
        raise RuntimeError("Phase 1 application currently supports only pgvector")
    engine, sessions = create_session_factory()
    repository = PostgresRepository(sessions)
    vectors = PgVectorStore(sessions)
    embedder = SentenceTransformersEmbedder(ingestion_config.embedding)
    chunker = (
        BlockAwareChunker(ingestion_config.chunker)
        if ingestion_config.chunker.type == "block_aware"
        else FixedWindowChunker(ingestion_config.chunker)
    )
    root = Path(os.environ.get("ARTIFACT_STORAGE_PATH", "./.artifacts"))
    ingestion = DocumentIngestionService(
        repository,
        LocalArtifactStore(root),
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
    reasoner = BasicGroundedReasoner(AqueductModelClient(architecture.reasoner.model))
    runner = BasicRagRunner(
        vector,
        summary,
        reasoner,
        SourceReferenceAssessor(),
        BasicGroundingPolicy(),
        GroundedAnswerRenderer(),
    )
    max_history = (
        architecture.chat_history.max_messages if architecture.chat_history else 10
    )
    return PhaseOneApplication(
        ProjectService(repository, ingestion),
        ChatService(repository, runner, max_history),
        engine,
    )
