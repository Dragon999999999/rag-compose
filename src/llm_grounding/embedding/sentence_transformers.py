"""Lazy Sentence Transformers adapter with cache-only query loading."""

import asyncio

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
        self._model: object | None = None

    @property
    def model_identity(self) -> str:
        """Return the configuration identity used by compatible vector indexes."""
        return self._config.identity()

    def _load_model(self, *, allow_download: bool) -> object:
        """Load one shared model, optionally allowing cache population for ingestion."""
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Install llm-grounding[embeddings] to use sentence_transformers"
            ) from exc
        self._model = SentenceTransformer(
            self._config.model,
            revision=self._config.revision,
            local_files_only=not allow_download,
        )
        return self._model

    def _encode(
        self, texts: list[str], *, query: bool, allow_download: bool
    ) -> tuple[EmbeddingVector, ...]:
        """Encode text synchronously and validate the configured dimension."""
        encoder = self._load_model(allow_download=allow_download)
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
        """Encode documents, allowing first-use cache population, off the event loop."""
        return await asyncio.to_thread(
            self._encode, texts, query=False, allow_download=True
        )

    async def embed_query(self, text: str) -> EmbeddingVector:
        """Encode one query from local weights without contacting the model hub."""
        return (
            await asyncio.to_thread(
                self._encode, [text], query=True, allow_download=False
            )
        )[0]
