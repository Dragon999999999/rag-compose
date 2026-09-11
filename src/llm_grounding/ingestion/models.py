"""Versioned ingestion choices, independent of query architecture configuration."""

from hashlib import sha256
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


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


class EmbeddingConfig(BaseModel):
    """Identify an embedding space, including normalization and optional revision."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    type: Literal["sentence_transformers", "fake"] = "sentence_transformers"
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    revision: str | None = None
    dimension: int = Field(default=384, ge=1, le=16000)
    normalize: bool = True
    batch_size: int = Field(default=32, ge=1, le=512)

    def identity(self) -> str:
        """Hash embedding semantics; batching does not change the embedding space."""
        content = self.model_dump_json(exclude={"batch_size"})
        return sha256(content.encode("utf-8")).hexdigest()


class IngestionConfig(BaseModel):
    """Record all effective representation choices for deduplication and rebuilds."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    parser: Literal["pymupdf"] = "pymupdf"
    parser_version: Literal["1"] = "1"
    knowledge_builder: Literal["basic"] = "basic"
    knowledge_version: Literal["1"] = "1"
    chunker: ChunkerConfig = Field(default_factory=ChunkerConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_backend: Literal["qdrant"] = "qdrant"
    index_version: Literal["1"] = "1"

    def identity(self) -> str:
        """Hash the full effective configuration, including component versions."""
        return sha256(self.model_dump_json().encode("utf-8")).hexdigest()
