"""Construct embedding providers from typed ingestion or architecture settings."""

from llm_grounding.embedding.fake import DeterministicFakeEmbedder
from llm_grounding.embedding.openai_compatible import OpenAICompatibleEmbedder
from llm_grounding.embedding.protocol import Embedder
from llm_grounding.embedding.sentence_transformers import SentenceTransformersEmbedder
from llm_grounding.infrastructure.config.models import (
    EmbedderConfig,
    LocalModelConfig,
    ModelConfig,
)
from llm_grounding.ingestion.models import EmbeddingConfig


def build_embedder(config: EmbeddingConfig) -> Embedder:
    """Build the registered embedder implementation selected by ``config.type``.

    This is the single construction boundary used by ingestion and query-time
    retrieval. Provider credentials are resolved by the concrete adapter only
    when that adapter is constructed.

    Raises:
        ValueError: If ``config`` selects an unsupported implementation.
    """
    if config.type == "sentence_transformers":
        return SentenceTransformersEmbedder(config)
    if config.type == "fake":
        return DeterministicFakeEmbedder(
            config.dimension, model_identity=config.identity()
        )
    if config.type == "openai_compatible":
        return OpenAICompatibleEmbedder(config)
    raise ValueError(f"Unsupported embedding type: {config.type}")


def build_architecture_embedder(config: EmbedderConfig) -> Embedder:
    """Build an architecture-level embedder without changing ingestion schemas.

    Architecture model settings intentionally omit an assumed vector dimension;
    the remote provider's response supplies it. Persisted ingestion indexes
    continue to use ``EmbeddingConfig`` and its explicit dimensionality.
    """
    if config.type == "openai_compatible":
        if not isinstance(config.model, ModelConfig):
            raise ValueError(
                "openai_compatible embedders require an OpenAI-compatible model"
            )
        return OpenAICompatibleEmbedder.from_settings(
            model=config.model.name,
            base_url=str(config.model.base_url),
            model_identity=config.identity(),
            timeout_seconds=config.model.timeout_seconds,
            retries=config.model.retries,
            batch_size=config.batch_size,
            api_key_env=config.model.api_key_env,
            api_key_file_env=config.model.api_key_file_env,
        )
    if config.type == "sentence_transformers":
        if not isinstance(config.model, LocalModelConfig):
            raise ValueError(
                "sentence_transformers embedders require a local model"
            )
        return SentenceTransformersEmbedder(
            EmbeddingConfig(
                type="sentence_transformers",
                model=config.model.name,
                revision=config.model.revision,
                dimension=config.model.dimension,
                normalize=config.model.normalize,
                batch_size=config.batch_size,
                cache_dir_env=config.model.cache_dir_env,
            )
        )
    raise ValueError(f"Unsupported embedding type: {config.type}")
