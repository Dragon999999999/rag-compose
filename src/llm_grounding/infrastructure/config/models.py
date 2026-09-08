"""Strict, credential-free architecture configuration at the application boundary.

Unknown fields are rejected. Component names are checked against the registry
before any provider is constructed. Provider and model settings stay outside
domain models; only sanitized JSON configuration is retained in run metadata.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


class ComponentConfig(BaseModel):
    """Select one registered implementation without arbitrary plugin loading."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    type: str = Field(
        min_length=1, description="Implementation name in the stage registry."
    )


class ModelConfig(BaseModel):
    """Configure Aqueduct requests; retries bound HTTP and output-validation retries."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    provider: Literal["aqueduct"]
    name: str = Field(
        min_length=1, description="Model identifier served by the endpoint."
    )
    base_url: HttpUrl
    timeout_seconds: float = Field(default=120, gt=0, le=3600, allow_inf_nan=False)
    retries: int = Field(default=2, ge=0, le=5)

    @field_validator("base_url")
    @classmethod
    def reject_url_credentials(cls, url: HttpUrl) -> HttpUrl:
        """Keep credentials and query tokens out of configuration and run records."""
        if url.username or url.password or url.query or url.fragment:
            raise ValueError(
                "base_url must not contain credentials, query, or fragment"
            )
        return url


class ReasonerConfig(ComponentConfig):
    """Select a reasoning implementation and its model configuration."""

    model: ModelConfig


class ArchitectureConfig(BaseModel):
    """Describe linear composition independently of the architecture's name."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    name: str = Field(min_length=1)
    version: int = Field(ge=0)
    parser: ComponentConfig
    knowledge_builder: ComponentConfig
    retriever: ComponentConfig
    reranker: ComponentConfig
    reasoner: ReasonerConfig
    assessors: tuple[ComponentConfig, ...]
    decision_policy: ComponentConfig
    renderer: ComponentConfig
