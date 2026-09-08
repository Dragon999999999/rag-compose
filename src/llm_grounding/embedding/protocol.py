"""Create explicit embeddings before indexing; vector stores never embed text.

Embedders may load local weights or call external services. Implementations must
return vectors in their declared space and preserve input order. Query vectors
must be compatible with document vectors from the selected ingestion.
"""

from typing import Protocol

from llm_grounding.embedding.models import EmbeddingVector


class Embedder(Protocol):
    """Create document and query vectors in one identifiable vector space."""

    @property
    def model_identity(self) -> str:
        """Return a stable identity used to reject incompatible indexes."""
        ...

    async def embed_documents(self, texts: list[str]) -> tuple[EmbeddingVector, ...]:
        """Embed document texts in input order, batching where appropriate."""
        ...

    async def embed_query(self, text: str) -> EmbeddingVector:
        """Embed a query using the compatible query encoding method."""
        ...
