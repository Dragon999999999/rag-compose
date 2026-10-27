"""Construct registered embedding providers from typed configuration."""

from collections.abc import Callable

from llm_grounding.embedding.fake import DeterministicFakeEmbedder
from llm_grounding.embedding.openai_compatible import OpenAICompatibleEmbedder
from llm_grounding.embedding.protocol import Embedder
from llm_grounding.embedding.sentence_transformers import (
    SentenceTransformersEmbedder,
)
from llm_grounding.ingestion.models import EmbeddingConfig

EmbeddingConstructor = Callable[[EmbeddingConfig], Embedder]


class EmbeddingProviderFactory:
    """Resolve embedding adapters through an explicit implementation registry."""

    def __init__(self) -> None:
        """Register the embedding providers shipped with the application."""
        self._constructors: dict[str, EmbeddingConstructor] = {
            "sentence_transformers": SentenceTransformersEmbedder,
            "fake": lambda config: DeterministicFakeEmbedder(
                config.dimension,
                model_identity=config.vector_identity(),
            ),
            "openai_compatible": OpenAICompatibleEmbedder,
        }

    def register(self, implementation: str, constructor: EmbeddingConstructor) -> None:
        """Register one provider implementation before runtime construction."""
        if not implementation.strip():
            raise ValueError("Embedding implementation name must be non-empty")
        self._constructors[implementation] = constructor

    def create(self, config: EmbeddingConfig) -> Embedder:
        """Construct the provider selected by a validated embedding configuration."""
        try:
            constructor = self._constructors[config.implementation]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported embedding implementation: {config.implementation}"
            ) from exc
        return constructor(config)

    @property
    def implementations(self) -> frozenset[str]:
        """Return registered provider identifiers for validation and diagnostics."""
        return frozenset(self._constructors)


_DEFAULT_FACTORY = EmbeddingProviderFactory()


def build_embedder(config: EmbeddingConfig) -> Embedder:
    """Build the registered embedder implementation selected by configuration.

    This is the single construction boundary used by ingestion and query-time
    retrieval. Provider credentials are resolved by the concrete adapter only
    when that adapter is constructed.

    Raises:
        ValueError: If ``config`` selects an unsupported implementation.
    """
    return _DEFAULT_FACTORY.create(config)
