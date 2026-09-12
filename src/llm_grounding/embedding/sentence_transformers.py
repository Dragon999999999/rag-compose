"""Lazy Sentence Transformers adapter with first-use model caching."""

import asyncio
from functools import cached_property

from llm_grounding.embedding.models import EmbeddingVector
from llm_grounding.ingestion.models import EmbeddingConfig


class SentenceTransformersEmbedder:
    """Encode document and query text with a configured sentence-transformer model."""

    def __init__(self, config: EmbeddingConfig) -> None:
        """Retain configuration and defer model loading until first use."""
        if config.type != "sentence_transformers":
            raise ValueError(
                "SentenceTransformersEmbedder requires sentence_transformers config"
            )
        self._config = config

    @property
    def model_identity(self) -> str:
        """Return the configuration identity used by compatible vector indexes."""
        return self._config.identity()

    @cached_property
    def _model(self) -> object:
        """Load cached weights or download them on first use."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Install llm-grounding[embeddings] to use sentence_transformers"
            ) from exc
        return SentenceTransformer(
            self._config.model, revision=self._config.revision, local_files_only=False
        )

    def _encode(self, texts: list[str], query: bool) -> tuple[EmbeddingVector, ...]:
        """Encode text synchronously and validate the configured dimension."""
        encoder = self._model
        method = encoder.encode_query if query else encoder.encode_document  # type: ignore[attr-defined]
        vectors = method(
            texts,
            batch_size=self._config.batch_size,
            normalize_embeddings=self._config.normalize,
        )
        output = tuple(
            EmbeddingVector(
                tuple(float(value) for value in vector), self.model_identity
            )
            for vector in vectors
        )
        if any(vector.dimension != self._config.dimension for vector in output):
            raise RuntimeError(
                "Embedding model dimension differs from configured dimension"
            )
        return output

    async def embed_documents(self, texts: list[str]) -> tuple[EmbeddingVector, ...]:
        """Encode documents in configured batches without blocking the event loop."""
        return await asyncio.to_thread(self._encode, texts, False)

    async def embed_query(self, text: str) -> EmbeddingVector:
        """Encode one query using the model's query-specific method when available."""
        return (await asyncio.to_thread(self._encode, [text], True))[0]
