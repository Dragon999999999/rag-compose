"""OpenAI-compatible embedding adapter for remote vector providers."""

from math import sqrt

from openai import AsyncOpenAI

from llm_grounding.embedding.models import EmbeddingVector
from llm_grounding.infrastructure.llm.credentials import resolve_embedding_key
from llm_grounding.ingestion.models import EmbeddingConfig


class OpenAICompatibleEmbedder:
    """Create embeddings through an OpenAI-compatible ``/embeddings`` API."""

    def __init__(self, config: EmbeddingConfig) -> None:
        """Create a client using sanitized configuration and call-time credentials."""
        if config.type != "openai_compatible":
            raise ValueError(
                "OpenAICompatibleEmbedder requires openai_compatible config"
            )
        if config.base_url is None:
            raise ValueError("OpenAI-compatible embeddings require base_url")
        self._config = config
        self._client = AsyncOpenAI(
            api_key=resolve_embedding_key(),
            base_url=str(config.base_url),
            timeout=config.timeout_seconds,
            max_retries=config.retries,
        )

    @property
    def model_identity(self) -> str:
        """Return the configuration identity used by compatible vector indexes."""
        return self._config.identity()

    async def _embed(self, texts: list[str]) -> tuple[EmbeddingVector, ...]:
        """Request a batch and validate order, cardinality, and dimensionality."""
        if not texts:
            return ()
        response = await self._client.embeddings.create(
            model=self._config.model,
            input=texts,
        )
        data = sorted(response.data, key=lambda item: item.index)
        if len(data) != len(texts):
            raise RuntimeError(
                "Embedding provider returned a different number of vectors"
            )
        vectors: list[EmbeddingVector] = []
        for item in data:
            values = [float(value) for value in item.embedding]
            if len(values) != self._config.dimension:
                raise RuntimeError(
                    "Embedding provider dimension differs from configured dimension"
                )
            if self._config.normalize:
                norm = sqrt(sum(value * value for value in values))
                if norm:
                    values = [value / norm for value in values]
            vectors.append(EmbeddingVector(tuple(values), self.model_identity))
        return tuple(vectors)

    async def embed_documents(self, texts: list[str]) -> tuple[EmbeddingVector, ...]:
        """Embed document texts while preserving provider response order."""
        return await self._embed(texts)

    async def embed_query(self, text: str) -> EmbeddingVector:
        """Embed one query in the same configured space as document vectors."""
        return (await self._embed([text]))[0]
