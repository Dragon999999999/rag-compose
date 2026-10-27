"""Versioned ingestion choices, independent of query architecture configuration."""

import json
from hashlib import sha256
from typing import Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    model_validator,
)


class ChunkerConfig(BaseModel):
    """Configure character, token, structural, or sentence-based chunking."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str | None = None
    type: Literal["chunker"] = "chunker"
    implementation: Literal[
        "fixed_window",
        "block_aware",
        "recursive_token",
        "hierarchical",
        "structural",
        "sentence_window",
        "semantic",
        "llm_guided",
        "late_chunking",
    ] = "block_aware"
    version: Literal["1"] = "1"
    target_chars: int = Field(default=800, ge=32, le=100000)
    overlap_chars: int = Field(default=120, ge=0)
    target_tokens: int = Field(default=256, ge=1, le=100000)
    max_tokens: int | None = Field(default=None, ge=1, le=100000)
    overlap_tokens: int = Field(default=32, ge=0, le=100000)
    tokenizer: Literal["whitespace", "unicode_words"] = "unicode_words"
    hierarchical_levels: tuple[int, ...] = (2048, 512, 128)
    sentences_per_chunk: int = Field(default=1, ge=1, le=100)
    sentence_window: int = Field(default=2, ge=0, le=100)
    model_id: str | None = Field(default=None, min_length=1)
    semantic_threshold: float = Field(default=0.25, ge=0, le=2)
    semantic_percentile: float | None = Field(default=None, ge=0, le=100)
    minimum_chunk_tokens: int = Field(default=1, ge=1, le=100000)
    llm_max_new_tokens: int = Field(default=256, ge=1, le=4096)

    @model_validator(mode="before")
    @classmethod
    def normalize_implementation_input(cls, value: object) -> object:
        """Normalize direct legacy type inputs without weakening YAML schemas."""
        if not isinstance(value, dict):
            return value
        selected = value.get("type")
        if (
            selected
            in {
                "fixed_window",
                "block_aware",
                "recursive_token",
                "hierarchical",
                "structural",
                "sentence_window",
                "semantic",
                "llm_guided",
                "late_chunking",
            }
            and "implementation" not in value
        ):
            normalized = dict(value)
            normalized["type"] = "chunker"
            normalized["implementation"] = selected
            return normalized
        return value

    @model_validator(mode="after")
    def validate_overlap(self) -> "ChunkerConfig":
        """Require progress between adjacent overlapping chunks."""
        if self.overlap_chars >= self.target_chars:
            raise ValueError("overlap_chars must be less than target_chars")
        maximum = self.max_tokens or self.target_tokens
        if self.max_tokens is not None and self.max_tokens < self.target_tokens:
            raise ValueError("max_tokens must be at least target_tokens")
        if self.overlap_tokens >= self.target_tokens:
            raise ValueError("overlap_tokens must be less than target_tokens")
        if not self.hierarchical_levels:
            raise ValueError("hierarchical_levels must not be empty")
        if any(level < 1 for level in self.hierarchical_levels):
            raise ValueError("hierarchical_levels must contain positive values")
        if any(
            parent <= child
            for parent, child in zip(
                self.hierarchical_levels, self.hierarchical_levels[1:], strict=False
            )
        ):
            raise ValueError("hierarchical_levels must be strictly descending")
        if self.overlap_tokens >= maximum:
            raise ValueError("overlap_tokens must be less than the token limit")
        if self.minimum_chunk_tokens > maximum:
            raise ValueError("minimum_chunk_tokens must not exceed the token limit")
        if self.implementation in {"semantic", "llm_guided", "late_chunking"}:
            if self.model_id is None:
                raise ValueError(
                    f"{self.implementation} chunking requires a registered model_id"
                )
        return self


class EmbeddingSpecification(BaseModel):
    """Describe the representation contract shared by query and document encoders."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    model: str = Field(min_length=1)
    revision: str | None = Field(default=None, min_length=1)
    embedding_space_id: str | None = Field(default=None, min_length=1)
    vector_kind: Literal["dense", "sparse", "multivector"] = "dense"
    dimensions: int = Field(ge=1, le=16000)
    normalization: bool = True
    similarity_metric: Literal["cosine", "dot", "euclidean"] = "cosine"
    query_prefix: str = ""
    document_prefix: str = ""


class EmbeddingConfig(BaseModel):
    """Identify an embedding space and its provider connection settings.

    API keys are deliberately absent. Providers resolve credentials at call time
    from the environment, while this model remains safe to persist with an
    ingestion record or include in an effective configuration.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    model_id: str | None = Field(default=None, min_length=1)
    location: Literal["local", "remote"] = "local"
    type: Literal[
        "embedding", "sentence_transformers", "fake", "openai_compatible"
    ] = (
        "sentence_transformers"
    )
    implementation: Literal["sentence_transformers", "fake", "openai_compatible"] = (
        "sentence_transformers"
    )
    provider: str | None = None
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_space_id: str | None = Field(default=None, min_length=1)
    representation: Literal["dense", "sparse", "multivector"] = Field(
        default="dense",
        validation_alias=AliasChoices("representation", "vector_kind"),
    )
    similarity_metric: Literal["cosine", "dot", "euclidean"] = "cosine"
    revision: str | None = None
    dimension: int = Field(
        default=384,
        ge=1,
        le=1_000_000,
        validation_alias=AliasChoices("dimension", "dimensions"),
    )
    normalize: bool = Field(
        default=True,
        validation_alias=AliasChoices("normalize", "normalization"),
    )
    batch_size: int = Field(default=32, ge=1, le=512)
    cache_dir_env: str | None = Field(default=None, min_length=1)
    query_prefix: str = ""
    document_prefix: str = ""
    trust_remote_code: bool = False
    late_chunking: bool = False
    base_url: HttpUrl | None = None
    api_key_env: str = "AQUEDUCT_API_KEY"
    api_key_file_env: str = "AQUEDUCT_API_KEY_FILE"
    timeout_seconds: float = Field(default=120, gt=0, le=3600, allow_inf_nan=False)
    retries: int = Field(default=2, ge=0, le=5)

    @model_validator(mode="before")
    @classmethod
    def normalize_provider_input(cls, value: object) -> object:
        """Normalize direct provider-style inputs into category plus implementation."""
        if not isinstance(value, dict):
            return value
        selected = value.get("type")
        if selected in {"sentence_transformers", "fake", "openai_compatible"}:
            normalized = dict(value)
            normalized["type"] = selected
            normalized.setdefault("implementation", selected)
            normalized.setdefault(
                "location", "remote" if selected == "openai_compatible" else "local"
            )
            return normalized
        return value

    @model_validator(mode="after")
    def validate_provider_settings(self) -> "EmbeddingConfig":
        """Require an endpoint for remote OpenAI-compatible embeddings."""
        if self.implementation == "openai_compatible" and self.location != "remote":
            raise ValueError("openai_compatible embeddings must be remote")
        if self.implementation != "openai_compatible" and self.location != "local":
            raise ValueError("local embedding implementations must be local")
        if self.implementation == "openai_compatible" and self.base_url is None:
            raise ValueError(
                "openai_compatible embeddings require a credential-free base_url"
            )
        if self.base_url is not None:
            if self.base_url.username or self.base_url.password:
                raise ValueError("base_url must not contain credentials")
            if self.base_url.query or self.base_url.fragment:
                raise ValueError("base_url must not contain query or fragment")
        return self

    def specification(self) -> EmbeddingSpecification:
        """Return provider-independent properties that define the vector space."""
        return EmbeddingSpecification(
            model=self.model,
            revision=self.revision,
            embedding_space_id=self.embedding_space_id,
            vector_kind=self.representation,
            dimensions=self.dimension,
            normalization=self.normalize,
            similarity_metric=self.similarity_metric,
            query_prefix=self.query_prefix,
            document_prefix=self.document_prefix,
        )

    def vector_identity(self) -> str:
        """Return the identity stored with vectors and used by index filtering.

        Legacy configurations retain their exact provider-aware identity. Explicit
        embedding-space declarations instead use only representation-defining
        properties, allowing compatible providers to share one index.
        """
        if self.embedding_space_id is None:
            return self.identity()
        content = json.dumps(
            self.specification().model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return sha256(content).hexdigest()

    def is_compatible_with(self, other: "EmbeddingConfig") -> bool:
        """Check whether two provider configurations can address one index."""
        if self.embedding_space_id is None or other.embedding_space_id is None:
            return self.identity() == other.identity()
        return self.specification() == other.specification()

    def is_compatible_with_spec(self, other: EmbeddingSpecification) -> bool:
        """Compare an explicitly declared provider against persisted space metadata."""
        if self.embedding_space_id is None or other.embedding_space_id is None:
            return False
        return self.specification() == other

    def identity(self) -> str:
        """Hash embedding semantics; transport tuning does not change space."""
        data = self.model_dump(
            mode="json",
            exclude={
                "batch_size",
                "cache_dir_env",
                "timeout_seconds",
                "retries",
                "api_key_env",
                "api_key_file_env",
                "model_id",
                "location",
            },
        )
        if self.embedding_space_id is None:
            data.pop("embedding_space_id", None)
        if self.representation == "dense":
            data.pop("representation", None)
        if self.similarity_metric == "cosine":
            data.pop("similarity_metric", None)
        if self.base_url is not None:
            data["base_url"] = sha256(str(self.base_url).encode("utf-8")).hexdigest()
        content = json.dumps(data, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return sha256(content).hexdigest()


class IngestionConfig(BaseModel):
    """Record all effective representation choices for deduplication and rebuilds."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    parser: Literal["pymupdf"] = "pymupdf"
    parser_version: Literal["1"] = "1"
    knowledge_builder: Literal[
        "basic", "structural", "metadata_rules", "metadata_spacy", "metadata_gliner"
    ] | None = "basic"
    knowledge_builder_config: dict[str, object] = Field(default_factory=dict)
    knowledge_version: Literal["1"] = "1"
    chunker: ChunkerConfig = Field(default_factory=ChunkerConfig)
    embedding: EmbeddingConfig | None = Field(default_factory=EmbeddingConfig)
    vector_backend: str | None = Field(default="qdrant", min_length=1)
    vector_backend_config: dict[str, object] = Field(default_factory=dict)
    index_version: Literal["1"] = "1"

    @model_validator(mode="after")
    def validate_dense_index_pair(self) -> "IngestionConfig":
        """Require embeddings and vector indexes to be enabled together."""
        if self.embedding is None and self.vector_backend is not None:
            raise ValueError(
                "vector_backend requires an embedding configuration; set both to null "
                "for PostgreSQL-only ingestion"
            )
        if self.embedding is not None and self.vector_backend is None:
            raise ValueError(
                "An embedding configuration requires a resolved vector_backend"
            )
        if self.vector_backend == "qdrant" and self.embedding is not None:
            if self.embedding.representation != "dense":
                raise ValueError("Qdrant supports dense embeddings only")
            if self.embedding.similarity_metric != "cosine":
                raise ValueError("Qdrant requires cosine similarity")
            if self.vector_backend_config.get("index_type", "flat") != "flat":
                raise ValueError("Qdrant does not support configurable index types")
        if self.embedding is not None and self.embedding.representation != "dense":
            raise ValueError(
                "Configured vector backends currently support dense embeddings only"
            )
        return self

    def identity(self) -> str:
        """Hash the full effective configuration, including component versions."""
        data = self.model_dump(mode="json")
        if not self.knowledge_builder_config:
            data.pop("knowledge_builder_config", None)
        embedding = data.get("embedding")
        if isinstance(embedding, dict) and self.embedding is not None:
            if self.embedding.embedding_space_id is not None:
                data["embedding"] = self.embedding.specification().model_dump(
                    mode="json"
                )
                embedding = data["embedding"]
            else:
                embedding.pop("embedding_space_id", None)
                if self.embedding.representation == "dense":
                    embedding.pop("representation", None)
                if self.embedding.similarity_metric == "cosine":
                    embedding.pop("similarity_metric", None)
        return sha256(
            json.dumps(data, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest()
