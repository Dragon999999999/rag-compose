"""Contract tests for persistent, lazy FAISS vector storage."""

from llm_grounding.domain.corpus import EvidenceReference, RetrievalUnit
from llm_grounding.embedding.models import EmbeddingVector
from llm_grounding.infrastructure.config.models import VectorBackendConfig
from llm_grounding.infrastructure.vector.faiss import FaissVectorStore


class ChunkRepository:
    """Hydrate canonical chunks for the derived FAISS adapter."""

    def __init__(self, chunks: tuple[RetrievalUnit, ...]) -> None:
        """Index canonical chunks by ID."""
        self._chunks = {chunk.chunk_id: chunk for chunk in chunks}

    async def chunks_by_ids(
        self, chunk_ids: tuple[str, ...]
    ) -> dict[str, RetrievalUnit]:
        """Return canonical chunks selected by the derived index."""
        return {chunk_id: self._chunks[chunk_id] for chunk_id in chunk_ids}


def _chunk(chunk_id: str, document_id: str, ingestion_id: str) -> RetrievalUnit:
    """Create a minimally complete canonical retrieval unit."""
    return RetrievalUnit(
        chunk_id,
        document_id,
        ingestion_id,
        f"{document_id}.pdf",
        chunk_id,
        0,
        1,
        1,
        (EvidenceReference(f"e-{chunk_id}", 0, 1),),
    )


async def test_faiss_is_lazy_persistent_and_project_scoped(tmp_path) -> None:
    """Flat FAISS indexes search, survive close, and preserve project isolation."""
    chunks = (
        _chunk("alpha", "doc-a", "ing-a"),
        _chunk("beta", "doc-b", "ing-b"),
    )
    repository = ChunkRepository(chunks)
    config = VectorBackendConfig(
        implementation="faiss", index_type="flat", idle_timeout_seconds=600
    )
    store = FaissVectorStore(config, repository, tmp_path)
    assert store.loaded_index_count == 0
    vector_a = EmbeddingVector((1.0, 0.0), "space")
    vector_b = EmbeddingVector((0.0, 1.0), "space")
    await store.upsert("project-a", chunks, (vector_a, vector_b))
    assert store.loaded_index_count == 1
    result = await store.search("project-a", vector_a, 1)
    assert result[0].chunk.chunk_id == "alpha"
    await store.close()
    assert store.loaded_index_count == 0
    restarted = FaissVectorStore(config, repository, tmp_path)
    result = await restarted.search("project-a", vector_a, 1)
    assert result[0].chunk.chunk_id == "alpha"
    assert restarted.loaded_index_count == 1
    assert not await restarted.search("project-b", vector_a, 1)


async def test_faiss_reuses_vectors_for_project_association(tmp_path) -> None:
    """Associating an ingestion copies persisted vectors without re-embedding."""
    chunk = _chunk("alpha", "doc-a", "ing-a")
    repository = ChunkRepository((chunk,))
    config = VectorBackendConfig(implementation="faiss", index_type="flat")
    store = FaissVectorStore(config, repository, tmp_path)
    vector = EmbeddingVector((1.0, 0.0), "space")
    await store.upsert("project-a", (chunk,), (vector,))
    assert await store.associate_ingestion("project-b", "ing-a")
    result = await store.search("project-b", vector, 1)
    assert [item.chunk.chunk_id for item in result] == ["alpha"]


async def test_faiss_approximate_index_types_are_supported(tmp_path) -> None:
    """HNSW and IVF indexes can be trained and queried through one contract."""
    chunks = tuple(
        _chunk(f"chunk-{index}", f"doc-{index}", "ing-a")
        for index in range(4)
    )
    vectors = tuple(
        EmbeddingVector((1.0 if index == 0 else 0.0, float(index)), "space")
        for index in range(4)
    )
    for index_type in ("hnsw", "ivf"):
        config = VectorBackendConfig(
            implementation="faiss",
            index_type=index_type,
            nlist=2,
            nprobe=2,
        )
        store = FaissVectorStore(config, ChunkRepository(chunks), tmp_path / index_type)
        await store.upsert("project", chunks, vectors)
        assert await store.search("project", vectors[0], 2)
