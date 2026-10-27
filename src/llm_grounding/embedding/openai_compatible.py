"""OpenAI-compatible embedding adapter for remote vector providers."""

from dataclasses import dataclass
from math import sqrt

from openai import AsyncOpenAI

from llm_grounding.embedding.models import EmbeddingVector
from llm_grounding.infrastructure.llm.credentials import resolve_api_key
from llm_grounding.ingestion.models import EmbeddingConfig


@dataclass(frozen=True)
class OpenAICompatibleEmbeddingSettings:
    """Provider-neutral settings used by architecture-level embedder config."""

    model: str
    base_url: str
    model_identity: str
    timeout_seconds: float = 120
    retries: int = 2
    batch_size: int = 32
    normalize: bool = True
    dimension: int | None = None
    api_key_env: str = "AQUEDUCT_API_KEY"
    api_key_file_env: str = "AQUEDUCT_API_KEY_FILE"


class OpenAICompatibleEmbedder:
    """Create embeddings through an OpenAI-compatible ``/embeddings`` API."""

    def __init__(
        self,
        config: EmbeddingConfig | OpenAICompatibleEmbeddingSettings,
    ) -> None:
        """Retain settings and defer client and credential creation to first use."""
        self._dimension: int | None
        if isinstance(config, EmbeddingConfig):
            if config.type != "openai_compatible":
                raise ValueError(
                    "OpenAICompatibleEmbedder requires openai_compatible config"
                )
            if config.base_url is None:
                raise ValueError("OpenAI-compatible embeddings require base_url")
            self._model = config.model
            self._base_url = str(config.base_url)
            self._model_identity = config.vector_identity()
            self._timeout_seconds = config.timeout_seconds
            self._retries = config.retries
            self._batch_size = config.batch_size
            self._normalize = config.normalize
            self._dimension = config.dimension
            self._api_key_env = config.api_key_env
            self._api_key_file_env = config.api_key_file_env
        else:
            if not config.model or not config.base_url or not config.model_identity:
                raise ValueError("OpenAI-compatible embedding settings are incomplete")
            if config.batch_size < 1:
                raise ValueError("Embedding batch_size must be positive")
            self._model = config.model
            self._base_url = config.base_url
            self._model_identity = config.model_identity
            self._timeout_seconds = config.timeout_seconds
            self._retries = config.retries
            self._batch_size = config.batch_size
            self._normalize = config.normalize
            self._dimension = config.dimension
            self._api_key_env = config.api_key_env
            self._api_key_file_env = config.api_key_file_env
        self._client: AsyncOpenAI | None = None

    @classmethod
    def from_settings(
        cls,
        *,
        model: str,
        base_url: str,
        model_identity: str,
        timeout_seconds: float = 120,
        retries: int = 2,
        batch_size: int = 32,
        normalize: bool = True,
        dimension: int | None = None,
        api_key_env: str = "AQUEDUCT_API_KEY",
        api_key_file_env: str = "AQUEDUCT_API_KEY_FILE",
    ) -> "OpenAICompatibleEmbedder":
        """Create an adapter from an architecture-level provider configuration."""
        return cls(
            OpenAICompatibleEmbeddingSettings(
                model=model,
                base_url=base_url,
                model_identity=model_identity,
                timeout_seconds=timeout_seconds,
                retries=retries,
                batch_size=batch_size,
                normalize=normalize,
                dimension=dimension,
                api_key_env=api_key_env,
                api_key_file_env=api_key_file_env,
            )
        )

    def _client_for_request(self) -> AsyncOpenAI:
        """Create one reusable client only when an embedding request is made."""
        if self._client is None:
            api_key = (
                resolve_api_key()
                if self._api_key_env == "AQUEDUCT_API_KEY"
                and self._api_key_file_env == "AQUEDUCT_API_KEY_FILE"
                else resolve_api_key(self._api_key_env, self._api_key_file_env)
            )
            self._client = AsyncOpenAI(
                api_key=api_key,
                base_url=self._base_url,
                timeout=self._timeout_seconds,
                max_retries=self._retries,
            )
        return self._client

    @property
    def model_identity(self) -> str:
        """Return the configuration identity used by compatible vector indexes."""
        return self._model_identity

    async def _embed_batch(self, texts: list[str]) -> tuple[EmbeddingVector, ...]:
        """Request a batch and validate order, cardinality, and dimensionality."""
        response = await self._client_for_request().embeddings.create(
            model=self._model,
            input=texts,
        )
        data = sorted(response.data, key=lambda item: item.index)
        if len(data) != len(texts):
            raise RuntimeError(
                "Embedding provider returned a different number of vectors"
            )
        if [item.index for item in data] != list(range(len(texts))):
            raise RuntimeError("Embedding provider returned invalid vector indexes")
        vectors: list[EmbeddingVector] = []
        for item in data:
            values = [float(value) for value in item.embedding]
            if self._dimension is not None and len(values) != self._dimension:
                raise RuntimeError(
                    "Embedding provider dimension differs from configured dimension"
                )
            if self._normalize:
                norm = sqrt(sum(value * value for value in values))
                if norm:
                    values = [value / norm for value in values]
            vectors.append(EmbeddingVector(tuple(values), self._model_identity))
        return tuple(vectors)

    async def _embed(self, texts: list[str]) -> tuple[EmbeddingVector, ...]:
        """Embed texts in bounded requests while preserving global input order."""
        if not texts:
            return ()
        vectors: list[EmbeddingVector] = []
        for start in range(0, len(texts), self._batch_size):
            vectors.extend(
                await self._embed_batch(texts[start : start + self._batch_size])
            )
        return tuple(vectors)

    async def embed_documents(self, texts: list[str]) -> tuple[EmbeddingVector, ...]:
        """Embed document texts while preserving provider response order."""
        return await self._embed(texts)

    async def embed_query(self, text: str) -> EmbeddingVector:
        """Embed one query in the same configured space as document vectors."""
        return (await self._embed([text]))[0]
