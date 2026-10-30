"""Regression tests for advanced semantic and architecture metric inputs."""

from datetime import UTC, datetime

from llm_grounding.application.eval_corpus_service import EvalCorpusService
from llm_grounding.domain.corpus import (
    Document,
    DocumentIngestion,
    EvidenceReference,
    IngestionStatus,
    RetrievalUnit,
)
from llm_grounding.domain.evidence import EvidenceType, EvidenceUnit
from llm_grounding.embedding.models import EmbeddingVector
from llm_grounding.ingestion.models import IngestionConfig
from llm_grounding.observability.ingestion import build_ingestion_observation


def test_embedding_identity_and_norm_are_stable() -> None:
    """Embedding diagnostics can compare vectors without persisting raw values."""
    first = EmbeddingVector((3.0, 4.0), "model-v1")
    second = EmbeddingVector((3.0, 4.0), "model-v1")

    assert first.vector_norm == 5.0
    assert first.embedding_id == second.embedding_id
    assert first.embedding_id != EmbeddingVector((3.0, 4.0), "model-v2").embedding_id


def test_ingestion_inventory_preserves_chunk_spans_and_counts() -> None:
    """Inventory records retain exact chunk text and evidence offsets."""
    now = datetime.now(UTC)
    document = Document("document", "paper.pdf", "a" * 64, "pdf/paper.pdf", now)
    ingestion = DocumentIngestion(
        "ingestion",
        document.id,
        "config",
        "{}",
        "embedding",
        2,
        IngestionStatus.READY,
        now,
    )
    evidence = (
        EvidenceUnit(
            "evidence",
            document.id,
            EvidenceType.TEXT,
            "file:///paper.pdf",
            0,
            page_number=1,
            text="source text",
        ),
    )
    chunks = (
        RetrievalUnit(
            "chunk",
            document.id,
            ingestion.id,
            document.name,
            "source",
            0,
            1,
            1,
            (EvidenceReference("evidence", 2, 8),),
        ),
    )

    observation = build_ingestion_observation(
        document,
        ingestion,
        IngestionConfig(),
        evidence,
        chunks,
        status="ready",
    )

    assert observation.chunk_count == 1
    assert observation.chunks[0].character_count == 6
    assert observation.chunks[0].source_spans == [
        {"evidence_id": "evidence", "start": 2, "end": 8}
    ]
    assert observation.extracted_units[0].page == 1


class MemoryEvalCorpusRepository:
    """Store corpus references without exposing production project methods."""

    def __init__(self) -> None:
        """Initialize an isolated in-memory corpus store."""
        self.corpora = {}
        self.operations = {}

    async def save_evaluation_corpus(self, corpus):
        """Persist only the isolated corpus reference."""
        self.corpora[corpus.corpus_id] = corpus

    async def get_evaluation_corpus(self, corpus_id):
        """Load one isolated corpus reference."""
        return self.corpora.get(corpus_id)

    async def delete_evaluation_corpus(self, corpus_id):
        """Delete only the isolated corpus namespace."""
        return self.corpora.pop(corpus_id, None) is not None

    async def save_evaluation_operation(self, operation_id, corpus_id, status, payload):
        """Persist a test operation without project state."""
        self.operations[operation_id] = payload

    async def get_evaluation_operation(self, operation_id):
        """Load a test operation."""
        return self.operations.get(operation_id)


async def test_eval_corpus_service_is_deterministic_and_isolated() -> None:
    """Derived benchmark corpora hash content and never call project persistence."""
    service = EvalCorpusService(MemoryEvalCorpusRepository())
    corpus = await service.create(
        corpus_id="eval-1",
        parent_corpus_id="production-project",
        documents=[{"id": "doc-1"}],
        perturbation_metadata={"operation": "typo"},
    )

    assert len(corpus.corpus_hash) == 64
    assert corpus.parent_corpus_id == "production-project"
    assert EvalCorpusService.deterministic_request_id(
        corpus_id="eval-1",
        benchmark_case_id="case-1",
        experiment_id="exp-1",
        query="question",
    ) == EvalCorpusService.deterministic_request_id(
        corpus_id="eval-1",
        benchmark_case_id="case-1",
        experiment_id="exp-1",
        query="question",
    )


async def test_eval_corpus_uploads_and_delete_are_isolated() -> None:
    """Controlled inventory operations never require production project storage."""
    repository = MemoryEvalCorpusRepository()
    service = EvalCorpusService(repository)
    await service.create(corpus_id="eval-2")

    document_operation = await service.upload_documents(
        "eval-2", [{"id": "doc-1", "name": "perturbed.pdf"}]
    )
    chunk_operation = await service.upload_chunks(
        "eval-2", [{"id": "chunk-1", "document_id": "doc-1", "text": "text"}]
    )

    assert document_operation["status"] == "completed"
    assert (await service.operation(chunk_operation["operation_id"]))[
        "operation"
    ] == "chunks"
    assert (await service.get("eval-2")).chunks[0]["id"] == "chunk-1"
    await service.delete("eval-2")
    assert repository.corpora == {}
