"""Regression tests for learned-vector Qdrant persistence."""

from unittest.mock import AsyncMock

from llm_grounding.domain.corpus import RetrievalUnit
from llm_grounding.embedding.models import TokenEmbedding
from llm_grounding.infrastructure.vector.learned import QdrantMultiVectorStore


class RecordingQdrantClient:
    """Record vector upload requests without contacting a Qdrant server."""

    def __init__(self) -> None:
        """Initialize an empty request log."""
        self.requests: list[list[object]] = []

    async def upsert(
        self,
        collection: str,
        points: list[object],
        *,
        wait: bool,
    ) -> None:
        """Record one server request and its completion mode."""
        assert collection == "test_multivector"
        assert wait
        self.requests.append(points)


def _chunk(chunk_id: str) -> RetrievalUnit:
    """Create a minimal canonical chunk for a derived-index test."""
    return RetrievalUnit(
        chunk_id,
        "document",
        "ingestion",
        "document.pdf",
        "text",
        0,
        1,
        1,
        (),
    )


async def test_multivector_upsert_batches_large_point_sets() -> None:
    """Large token indexes are split instead of exceeding Qdrant's request limit."""
    client = RecordingQdrantClient()
    store = object.__new__(QdrantMultiVectorStore)
    store._collection = "test_multivector"
    store._model_identity = "colbert-v2"
    store._client = client  # type: ignore[assignment]
    store._ensure_collection = AsyncMock()

    chunks = tuple(_chunk(str(index)) for index in range(17))
    vectors = tuple(
        TokenEmbedding(((1.0, 0.0),), "colbert-v2") for _ in chunks
    )

    await store.upsert("project", chunks, vectors)

    assert [len(request) for request in client.requests] == [8, 8, 1]
