"""Derived vector index contract with project filtering at search time.

Canonical chunks and provenance are persisted separately. A vector store can be
rebuilt from them. Searches must apply the project filter in the index operation,
not retrieve globally and filter in application memory.
"""

from typing import Protocol

from llm_grounding.domain.corpus import ChunkRetrievalResult, RetrievalUnit
from llm_grounding.embedding.models import EmbeddingVector


class VectorStore(Protocol):
    """Index derived chunk vectors and search them within a project corpus."""

    async def upsert(
        self,
        project_id: str,
        chunks: tuple[RetrievalUnit, ...],
        vectors: tuple[EmbeddingVector, ...],
    ) -> None:
        """Upsert equally sized chunk/vector batches for one project association."""
        ...

    async def delete_ingestion(self, ingestion_id: str) -> None:
        """Remove derived vectors for an ingestion without deleting canonical records."""

    async def associate_ingestion(self, project_id: str, ingestion_id: str) -> None:
        """Associate an existing ingestion's derived vectors with another project.

        Implementations must reuse the already-generated vectors and must not invoke
        an embedder. The association allows a content-addressed document ingestion to
        be selected by more than one project while retaining index-time project scope.
        """
        ...

    async def search(
        self,
        project_id: str,
        query: EmbeddingVector,
        top_k: int,
        document_ids: tuple[str, ...] = (),
    ) -> tuple[ChunkRetrievalResult, ...]:
        """Return score-ranked chunks scoped by project and optional documents."""
        ...
