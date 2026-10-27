"""Embedding adapter backed by a lease from the local model manager."""

from contextlib import AbstractAsyncContextManager
from typing import Protocol

from llm_grounding.embedding.models import EmbeddingVector
from llm_grounding.ingestion.models import EmbeddingConfig


class _LocalModelLease(Protocol):
    """Subset of the local model manager used by the embedding adapter."""

    async def embed(self, texts: list[str]) -> tuple[tuple[float, ...], ...]:
        """Embed texts through a ready local runtime."""
        ...


class _LocalModelManager(Protocol):
    """Acquire a local model without coupling embedding to Docker."""

    def acquire(self, model_id: str) -> AbstractAsyncContextManager[_LocalModelLease]:
        """Return an async context manager yielding a local model lease."""
        ...


class ManagedLocalEmbedder:
    """Implement the existing embedder contract using LocalModelManager leases."""

    def __init__(self, config: EmbeddingConfig, manager: _LocalModelManager) -> None:
        """Retain model identity and defer storage/runtime work to first use."""
        if config.location != "local" or config.model_id is None:
            raise ValueError("ManagedLocalEmbedder requires a local model_id")
        self._config = config
        self._manager = manager

    @property
    def model_identity(self) -> str:
        """Return the configured identity used by vector-index compatibility checks."""
        return self._config.vector_identity()

    async def embed_documents(self, texts: list[str]) -> tuple[EmbeddingVector, ...]:
        """Acquire the local runtime for one document batch."""
        return await self._embed(texts, query=False)

    async def embed_query(self, text: str) -> EmbeddingVector:
        """Acquire the local runtime for one query and preserve its vector space."""
        return (await self._embed([text], query=True))[0]

    async def _embed(
        self, texts: list[str], *, query: bool
    ) -> tuple[EmbeddingVector, ...]:
        """Translate a local server response into the shared embedding contract."""
        if not texts:
            return ()
        assert self._config.model_id is not None
        prefix = self._config.query_prefix if query else self._config.document_prefix
        async with self._manager.acquire(self._config.model_id) as model:
            values = await model.embed([f"{prefix}{text}" for text in texts])
        if len(values) != len(texts):
            raise RuntimeError(
                "Local embedding server returned an unexpected vector count"
            )
        vectors = tuple(
            EmbeddingVector(tuple(vector), self.model_identity) for vector in values
        )
        if any(vector.dimension != self._config.dimension for vector in vectors):
            raise RuntimeError(
                "Local embedding server dimension differs from configuration"
            )
        return vectors
