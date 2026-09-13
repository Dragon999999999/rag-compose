"""Offline contract tests for Qdrant payload filtering and canonical hydration."""

from types import SimpleNamespace

from qdrant_client import models

from llm_grounding.domain.corpus import EvidenceReference, RetrievalUnit
from llm_grounding.embedding.models import EmbeddingVector
from llm_grounding.infrastructure.vector.qdrant import QdrantVectorStore


class FakeQdrantClient:
    """Minimal async Qdrant substitute that evaluates adapter payload predicates."""

    def __init__(self) -> None:
        """Start with no collection and no derived points."""
        self.exists = False
        self.points: dict[str, models.PointStruct] = {}
        self.info: object | None = None

    async def collection_exists(self, collection: str) -> bool:
        """Report whether the adapter has initialized its collection."""
        return self.exists

    async def create_collection(
        self, collection: str, vectors_config: models.VectorParams
    ) -> None:
        """Remember collection parameters for compatibility validation."""
        self.exists = True
        self.info = SimpleNamespace(
            config=SimpleNamespace(params=SimpleNamespace(vectors=vectors_config))
        )

    async def create_payload_index(self, *args: object, **kwargs: object) -> None:
        """Accept adapter indexing calls; indexes do not affect fake matching."""

    async def get_collection(self, collection: str) -> object:
        """Return the stored collection metadata."""
        assert self.info is not None
        return self.info

    async def upsert(
        self, collection: str, points: list[models.PointStruct], wait: bool
    ) -> None:
        """Store replacement points by their deterministic adapter IDs."""
        self.points.update({str(point.id): point for point in points})

    async def scroll(
        self,
        collection: str,
        scroll_filter: models.Filter,
        **_: object,
    ) -> tuple[list[models.PointStruct], None]:
        """Return payload-matching points for reusable-ingestion association."""
        return self._matching(scroll_filter), None

    async def set_payload(
        self,
        collection: str,
        payload: dict[str, object],
        points: list[str],
        **_: object,
    ) -> None:
        """Update just the requested payload fields on selected points."""
        for point_id in points:
            point = self.points[point_id]
            point.payload = {**(point.payload or {}), **payload}

    async def delete(
        self,
        collection: str,
        points_selector: models.Filter,
        **_: object,
    ) -> None:
        """Remove all points selected by the adapter's derived-state filter."""
        for point in self._matching(points_selector):
            del self.points[str(point.id)]

    async def query_points(
        self,
        collection: str,
        query: list[float],
        query_filter: models.Filter,
        **_: object,
    ) -> object:
        """Return matching points in deterministic order with a synthetic score."""
        points = [
            SimpleNamespace(id=point.id, payload=point.payload, score=0.9)
            for point in self._matching(query_filter)
        ]
        return SimpleNamespace(points=points)

    def _matching(self, filter_: models.Filter) -> list[models.PointStruct]:
        """Evaluate the MatchValue and MatchAny predicates used by this adapter."""
        output: list[models.PointStruct] = []
        for point in self.points.values():
            payload = point.payload or {}
            matches = True
            for condition in filter_.must or []:
                assert isinstance(condition, models.FieldCondition)
                value = payload.get(condition.key)
                match = condition.match
                if isinstance(match, models.MatchValue):
                    matches = matches and (
                        match.value in value
                        if isinstance(value, list)
                        else value == match.value
                    )
                elif isinstance(match, models.MatchAny):
                    matches = matches and (
                        bool(set(value) & set(match.any))
                        if isinstance(value, list)
                        else value in match.any
                    )
            if matches:
                output.append(point)
        return output

    async def close(self) -> None:
        """Match the real client's close method."""


class ChunkRepository:
    """Hydrate test chunks in the same manner as the PostgreSQL repository boundary."""

    def __init__(self, chunks: tuple[RetrievalUnit, ...]) -> None:
        """Index canonical chunks by identifier."""
        self._chunks = {chunk.chunk_id: chunk for chunk in chunks}

    async def chunks_by_ids(
        self, chunk_ids: tuple[str, ...]
    ) -> dict[str, RetrievalUnit]:
        """Return only canonical chunks requested by the derived index."""
        return {chunk_id: self._chunks[chunk_id] for chunk_id in chunk_ids}


def _chunk(chunk_id: str, document_id: str, ingestion_id: str) -> RetrievalUnit:
    """Create a canonical chunk with enough provenance for hydration assertions."""
    return RetrievalUnit(
        chunk_id,
        document_id,
        ingestion_id,
        f"{document_id}.pdf",
        f"Text for {chunk_id}",
        0,
        1,
        1,
        (EvidenceReference(f"e-{chunk_id}", 0, 1),),
    )


async def test_qdrant_search_filters_project_before_canonical_hydration() -> None:
    """Project A cannot retrieve a globally similar vector belonging solely to B."""
    alpha = _chunk("alpha", "document-a", "ingestion-a")
    beta = _chunk("beta", "document-b", "ingestion-b")
    fake = FakeQdrantClient()
    # Avoid constructing the real client: ordinary unit tests must not attempt
    # any network connection or client/server compatibility check.
    store = object.__new__(QdrantVectorStore)
    store._collection = "test"
    store._repository = ChunkRepository((alpha, beta))  # type: ignore[assignment]
    store._client = fake  # type: ignore[assignment]
    vector = EmbeddingVector((1.0, 0.0), "test-space")
    await store.upsert("project-a", (alpha,), (vector,))
    await store.upsert("project-b", (beta,), (vector,))

    results = await store.search("project-a", vector, 5)

    assert [result.chunk.chunk_id for result in results] == ["alpha"]
    assert results[0].chunk.text == "Text for alpha"
    await store.associate_ingestion("project-c", "ingestion-a")
    assert [
        result.chunk.chunk_id for result in await store.search("project-c", vector, 5)
    ] == ["alpha"]
    await store.remove_project("project-a")
    assert not await store.search("project-a", vector, 5)
    assert [
        result.chunk.chunk_id for result in await store.search("project-c", vector, 5)
    ] == ["alpha"]
    await store.delete_ingestion("ingestion-a")
    assert not await store.search("project-c", vector, 5)


async def test_qdrant_association_reports_missing_ingestion_vectors() -> None:
    """An absent collection or empty ingestion is a rebuild signal, not an error."""
    fake = FakeQdrantClient()
    store = object.__new__(QdrantVectorStore)
    store._collection = "test"
    store._repository = ChunkRepository(())  # type: ignore[assignment]
    store._client = fake  # type: ignore[assignment]

    assert not await store.associate_ingestion("project", "ingestion")
    fake.exists = True
    assert not await store.associate_ingestion("project", "ingestion")
