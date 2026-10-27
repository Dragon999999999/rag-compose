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
    """Size chunks by Unicode characters, with overlap strictly below target size."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    type: Literal["fixed_window", "block_aware"] = "block_aware"
    version: Literal["1"] = "1"
    target_chars: int = Field(default=800, ge=32, le=100000)
    overlap_chars: int = Field(default=120, ge=0)

    @model_validator(mode="after")
    def validate_overlap(self) -> "ChunkerConfig":
        """Require progress between adjacent overlapping chunks."""
        if self.overlap_chars >= self.target_chars:
            raise ValueError("overlap_chars must be less than target_chars")
        return self


class EmbeddingSpecification(BaseModel):
    """Describe the representation contract shared by query and document encoders."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    model: str = Field(min_length=1)
    revision: str | None = Field(default=None, min_length=1)
    embedding_space_id: str | None = Field(default=None, min_length=1)
    vector_kind: Literal["dense", "sparse"] = "dense"
    dimensions: int = Field(ge=1, le=16000)
    normalization: bool = True
    similarity_metric: Literal["cosine", "dot", "euclidean"] = "cosine"


class EmbeddingConfig(BaseModel):
    """Identify an embedding space and its provider connection settings.

    API keys are deliberately absent. Providers resolve credentials at call time
    from the environment, while this model remains safe to persist with an
    ingestion record or include in an effective configuration.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    type: Literal["sentence_transformers", "fake", "openai_compatible"] = (
        "sentence_transformers"
    )
    provider: str | None = None
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_space_id: str | None = Field(default=None, min_length=1)
    representation: Literal["dense", "sparse"] = Field(
        default="dense",
        validation_alias=AliasChoices("representation", "vector_kind"),
    )
    similarity_metric: Literal["cosine", "dot", "euclidean"] = "cosine"
    revision: str | None = None
    dimension: int = Field(
        default=384,
        ge=1,
        le=16000,
        validation_alias=AliasChoices("dimension", "dimensions"),
    )
    normalize: bool = Field(
        default=True,
        validation_alias=AliasChoices("normalize", "normalization"),
    )
    batch_size: int = Field(default=32, ge=1, le=512)
    cache_dir_env: str | None = Field(default=None, min_length=1)
    base_url: HttpUrl | None = None
    api_key_env: str = "AQUEDUCT_API_KEY"
    api_key_file_env: str = "AQUEDUCT_API_KEY_FILE"
    timeout_seconds: float = Field(default=120, gt=0, le=3600, allow_inf_nan=False)
    retries: int = Field(default=2, ge=0, le=5)

    @model_validator(mode="after")
    def validate_provider_settings(self) -> "EmbeddingConfig":
        """Require an endpoint for remote OpenAI-compatible embeddings."""
        if self.type == "openai_compatible" and self.base_url is None:
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
    knowledge_builder: Literal["basic"] | None = "basic"
    knowledge_version: Literal["1"] = "1"
    chunker: ChunkerConfig = Field(default_factory=ChunkerConfig)
    embedding: EmbeddingConfig | None = Field(default_factory=EmbeddingConfig)
    vector_backend: str | None = Field(default="qdrant", min_length=1)
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
        return self

    def identity(self) -> str:
        """Hash the full effective configuration, including component versions."""
        data = self.model_dump(mode="json")
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
            json.dumps(data, separators=(",", ":"), ensure_ascii=False).encode(
                "utf-8"
            )
        ).hexdigest()
