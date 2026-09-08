"""Deterministic in-memory vector index for tests; never a canonical persistence store."""

from dataclasses import dataclass

from llm_grounding.domain.corpus import ChunkRetrievalResult, RetrievalUnit
from llm_grounding.embedding.models import EmbeddingVector


@dataclass(frozen=True)
class _IndexedChunk:
    """Keep a derived chunk with its vector and owning project association."""

    project_id: str
    chunk: RetrievalUnit
    vector: EmbeddingVector


class InMemoryVectorStore:
    """Provide exact cosine similarity with index-time project metadata filtering."""

    def __init__(self) -> None:
        """Start with an empty derived index."""
        self._items: dict[tuple[str, str], _IndexedChunk] = {}

    async def upsert(
        self,
        project_id: str,
        chunks: tuple[RetrievalUnit, ...],
        vectors: tuple[EmbeddingVector, ...],
    ) -> None:
        """Store a batch after checking cardinality and embedding-space compatibility."""
        if len(chunks) != len(vectors):
            raise ValueError("Each retrieval chunk needs exactly one vector")
        for chunk, vector in zip(chunks, vectors, strict=True):
            self._items[(project_id, chunk.chunk_id)] = _IndexedChunk(
                project_id, chunk, vector
            )

    async def delete_ingestion(self, ingestion_id: str) -> None:
        """Remove every derived index entry produced by one ingestion."""
        keys = [
            key
            for key, item in self._items.items()
            if item.chunk.ingestion_id == ingestion_id
        ]
        for key in keys:
            del self._items[key]

    async def associate_ingestion(self, project_id: str, ingestion_id: str) -> None:
        """Copy existing derived-vector associations without re-embedding content."""
        matches = [
            item
            for item in self._items.values()
            if item.chunk.ingestion_id == ingestion_id
        ]
        if not matches:
            raise KeyError(f"No indexed vectors for ingestion: {ingestion_id}")
        for item in matches:
            self._items[(project_id, item.chunk.chunk_id)] = _IndexedChunk(
                project_id,
                item.chunk,
                item.vector,
            )

    async def search(
        self,
        project_id: str,
        query: EmbeddingVector,
        top_k: int,
        document_ids: tuple[str, ...] = (),
    ) -> tuple[ChunkRetrievalResult, ...]:
        """Search only matching project/document records with exact cosine similarity."""
        if top_k < 1:
            raise ValueError("top_k must be positive")
        candidates: list[tuple[float, RetrievalUnit]] = []
        for item in self._items.values():
            if item.project_id != project_id or (
                document_ids and item.chunk.document_id not in document_ids
            ):
                continue
            if (
                item.vector.model_identity != query.model_identity
                or item.vector.dimension != query.dimension
            ):
                raise ValueError(
                    "Query embedding is incompatible with indexed chunk embeddings"
                )
            score = sum(
                left * right
                for left, right in zip(item.vector.values, query.values, strict=True)
            )
            candidates.append((score, item.chunk))
        candidates.sort(key=lambda item: (-item[0], item[1].chunk_id))
        return tuple(
            ChunkRetrievalResult(chunk, rank, "vector", score)
            for rank, (score, chunk) in enumerate(candidates[:top_k], 1)
        )
