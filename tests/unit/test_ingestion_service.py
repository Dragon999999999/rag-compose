"""Test deduplicated ingestion reuse and derived-vector recovery."""

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from llm_grounding.domain.corpus import (
    Document,
    DocumentIngestion,
    IngestionStatus,
    ProjectDocument,
)
from llm_grounding.domain.documents import (
    DocumentElement,
    DocumentSource,
    ParsedDocument,
)
from llm_grounding.embedding.models import EmbeddingVector
from llm_grounding.infrastructure.vector.memory import InMemoryVectorStore
from llm_grounding.ingestion.models import (
    ChunkerConfig,
    EmbeddingConfig,
    IngestionConfig,
)
from llm_grounding.ingestion.service import (
    DocumentIngestionService,
    _is_generated_pdf_name,
)
from llm_grounding.knowledge.basic import BasicKnowledgeBuilder
from llm_grounding.knowledge.chunking.fixed_window import FixedWindowChunker


class MatchingEmbedder:
    """Produce deterministic vectors in the ingestion's configured space."""

    def __init__(self, model_identity: str, dimension: int) -> None:
        """Configure the vector identity and dimensionality."""
        self.model_identity = model_identity
        self._dimension = dimension

    async def embed_documents(self, texts: list[str]) -> tuple[EmbeddingVector, ...]:
        """Return one compatible vector for each supplied chunk."""
        values = (1.0,) + (0.0,) * (self._dimension - 1)
        return tuple(EmbeddingVector(values, self.model_identity) for _ in texts)

    async def embed_query(self, text: str) -> EmbeddingVector:
        """Return a compatible query vector for protocol completeness."""
        values = (1.0,) + (0.0,) * (self._dimension - 1)
        return EmbeddingVector(values, self.model_identity)


class RecordingVectorStore(InMemoryVectorStore):
    """Record reuse checks while retaining the in-memory derived index."""

    def __init__(self) -> None:
        """Start empty so the simulated derived vectors are missing."""
        super().__init__()
        self.association_attempts: list[tuple[str, str]] = []

    async def associate_ingestion(self, project_id: str, ingestion_id: str) -> bool:
        """Record and perform one reusable-vector association attempt."""
        self.association_attempts.append((project_id, ingestion_id))
        return await super().associate_ingestion(project_id, ingestion_id)


class FixedDigestIngestionService(DocumentIngestionService):
    """Avoid filesystem hashing in the focused deduplication unit test."""

    async def _digest(self, source: Path) -> str:
        """Return the digest represented by the existing document record."""
        return "stored-digest"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("tmpjluizqrw.pdf", True),
        ("64710113-74ab-4d84-98a9-ab70cf5d169e.pdf", True),
        (f"{'a' * 64}.pdf", True),
        ("tmp_report.pdf", False),
        ("Kumpf2024.pdf", False),
    ],
)
def test_generated_pdf_name_detection(name: str, expected: bool) -> None:
    """Recognize infrastructure names without classifying ordinary filenames."""
    assert _is_generated_pdf_name(name) is expected


async def test_reuse_repairs_legacy_temporary_document_name(tmp_path: Path) -> None:
    """Reuse content while replacing legacy upload names in canonical provenance."""
    source = tmp_path / "Original Research Paper.pdf"
    source.write_bytes(b"%PDF-1.7 test")
    config = IngestionConfig(
        chunker=ChunkerConfig(target_chars=64, overlap_chars=0),
        embedding=EmbeddingConfig(type="fake", model="test", dimension=2),
    )
    document = Document(
        "document",
        "tmpjluizqrw.pdf",
        "stored-digest",
        "pdf/stored-digest.pdf",
        datetime.now(UTC),
    )
    repaired = replace(document, name=source.name)
    ingestion = DocumentIngestion(
        "ingestion",
        document.id,
        config.identity(),
        config.model_dump_json(),
        config.embedding.identity(),
        config.embedding.dimension,
        IngestionStatus.READY,
        datetime.now(UTC),
    )
    repository = SimpleNamespace(
        find_document_by_sha256=AsyncMock(return_value=document),
        update_document_name=AsyncMock(return_value=repaired),
        find_ingestion=AsyncMock(return_value=ingestion),
        create_ingestion=AsyncMock(
            side_effect=AssertionError("Ready ingestion should be reused")
        ),
        attach_document=AsyncMock(
            return_value=ProjectDocument("project", document.id, ingestion.id)
        ),
    )
    artifacts = SimpleNamespace(
        put_pdf=AsyncMock(
            side_effect=AssertionError("Existing document must reuse its artifact")
        )
    )
    vectors = SimpleNamespace(associate_ingestion=AsyncMock(return_value=True))
    service = FixedDigestIngestionService(
        repository,  # type: ignore[arg-type]
        artifacts,  # type: ignore[arg-type]
        SimpleNamespace(),  # type: ignore[arg-type]
        BasicKnowledgeBuilder(),
        FixedWindowChunker(config.chunker),
        MatchingEmbedder(config.embedding.identity(), config.embedding.dimension),
        vectors,  # type: ignore[arg-type]
    )

    result, _, result_ingestion = await service.add_document("project", source, config)

    assert result.name == source.name
    assert result_ingestion is ingestion
    repository.update_document_name.assert_awaited_once_with(document.id, source.name)
    vectors.associate_ingestion.assert_awaited_once_with("project", ingestion.id)


@pytest.mark.parametrize(
    "status",
    [IngestionStatus.READY, IngestionStatus.FAILED, IngestionStatus.INDEXING],
)
async def test_existing_ingestion_without_vectors_is_rebuilt(
    tmp_path: Path, status: IngestionStatus
) -> None:
    """Retry PostgreSQL records that have no matching derived vectors."""
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"%PDF-1.7 test")
    config = IngestionConfig(
        chunker=ChunkerConfig(target_chars=64, overlap_chars=0),
        embedding=EmbeddingConfig(type="fake", model="test", dimension=2),
    )
    document = Document(
        "document",
        source.name,
        "stored-digest",
        "pdf/stored-digest.pdf",
        datetime.now(UTC),
    )
    ingestion = DocumentIngestion(
        "ingestion",
        document.id,
        config.identity(),
        config.model_dump_json(),
        config.embedding.identity(),
        config.embedding.dimension,
        status,
        datetime.now(UTC),
    )
    refreshed = replace(ingestion, status=IngestionStatus.READY, error=None)
    repository = SimpleNamespace(
        find_document_by_sha256=AsyncMock(return_value=document),
        find_ingestion=AsyncMock(side_effect=[ingestion, refreshed]),
        create_ingestion=AsyncMock(
            side_effect=AssertionError("Existing ingestion should be retried")
        ),
        attach_document=AsyncMock(
            return_value=ProjectDocument("project", document.id, ingestion.id)
        ),
        clear_ingestion_representation=AsyncMock(),
        set_ingestion_status=AsyncMock(),
        store_evidence=AsyncMock(),
        store_chunks=AsyncMock(),
        project_ids_for_ingestion=AsyncMock(return_value=("project",)),
    )
    artifacts = SimpleNamespace(
        put_pdf=AsyncMock(
            side_effect=AssertionError("Existing document must reuse its artifact")
        ),
        path_for=AsyncMock(return_value=source),
    )
    parsed = ParsedDocument(
        DocumentSource(document.id, source),
        (
            DocumentElement(
                document.id,
                1,
                0,
                0,
                "Grounded content that must be rebuilt into the vector index.",
            ),
        ),
        1,
    )
    parser = SimpleNamespace(parse=AsyncMock(return_value=parsed))
    embedder = MatchingEmbedder(config.embedding.identity(), config.embedding.dimension)
    vectors = RecordingVectorStore()
    service = FixedDigestIngestionService(
        repository,  # type: ignore[arg-type]
        artifacts,  # type: ignore[arg-type]
        parser,  # type: ignore[arg-type]
        BasicKnowledgeBuilder(),
        FixedWindowChunker(config.chunker),
        embedder,
        vectors,
    )

    _, association, result = await service.add_document("project", source, config)

    assert association.ingestion_id == ingestion.id
    assert result.status == IngestionStatus.READY
    repository.clear_ingestion_representation.assert_awaited_once_with(ingestion.id)
    repository.store_chunks.assert_awaited_once()
    parser.parse.assert_awaited_once()
    assert vectors.association_attempts == (
        [("project", ingestion.id)] if status == IngestionStatus.READY else []
    )
    indexed = await vectors.search(
        "project", await embedder.embed_query("rebuilt vector index"), 10
    )
    assert indexed
    assert all(item.chunk.ingestion_id == ingestion.id for item in indexed)
