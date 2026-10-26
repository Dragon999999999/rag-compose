"""Construct embedding providers from typed ingestion or architecture settings."""

from llm_grounding.embedding.fake import DeterministicFakeEmbedder
from llm_grounding.embedding.openai_compatible import OpenAICompatibleEmbedder
from llm_grounding.embedding.protocol import Embedder
from llm_grounding.embedding.sentence_transformers import SentenceTransformersEmbedder
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
