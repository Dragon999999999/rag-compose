"""Typed standalone model, component, and pipeline configuration sources."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class CredentialReference(BaseModel):
    """Name environment variables that provide a provider credential at runtime."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    env_var: str = Field(default="AQUEDUCT_API_KEY", min_length=1)
    file_env_var: str = Field(default="AQUEDUCT_API_KEY_FILE", min_length=1)


class ModelDefinitionBase(BaseModel):
    """Common identity and provider fields for standalone model definitions."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    id: str = Field(
        min_length=1,
        pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$",
        description="Stable internal model-configuration identifier.",
    )
    stage: Literal["embedding", "reasoner", "reranker"] = Field(
        description="Pipeline stage allowed to reference this model."
    )
    name: str = Field(
        min_length=1,
        description="Provider or local runtime model name; not the internal ID.",
    )


class OpenAICompatibleModelDefinition(ModelDefinitionBase):
    """Configure a model served through an OpenAI-compatible HTTP endpoint."""

    backend: Literal["openai_compatible"]
    provider: Literal["aqueduct"] = "aqueduct"
    base_url: HttpUrl
    auth: CredentialReference = Field(default_factory=CredentialReference)
    timeout_seconds: float = Field(default=120, gt=0, le=3600, allow_inf_nan=False)
    retries: int = Field(default=2, ge=0, le=5)
    dimension: int | None = Field(default=None, ge=1, le=16000)
    embedding_space_id: str | None = Field(default=None, min_length=1)
    representation: Literal["dense", "sparse"] = "dense"
    similarity_metric: Literal["cosine", "dot", "euclidean"] = "cosine"

    @model_validator(mode="after")
    def reject_url_credentials(self) -> "OpenAICompatibleModelDefinition":
        """Keep credentials and query tokens out of endpoint configuration."""
        if (
            self.base_url.username
            or self.base_url.password
            or self.base_url.query
            or self.base_url.fragment
        ):
            raise ValueError(
                "base_url must not contain credentials, query, or fragment"
            )
        return self


class SentenceTransformersModelDefinition(ModelDefinitionBase):
    """Configure a locally loaded Sentence Transformers model."""

    backend: Literal["sentence_transformers"]
    revision: str | None = None
    dimension: int = Field(default=384, ge=1, le=16000)
    normalize: bool = True
    batch_size: int = Field(default=32, ge=1, le=512)
    cache_dir_env: str | None = Field(default=None, min_length=1)
    embedding_space_id: str | None = Field(default=None, min_length=1)
    representation: Literal["dense", "sparse"] = "dense"
    similarity_metric: Literal["cosine", "dot", "euclidean"] = "cosine"


class FakeModelDefinition(ModelDefinitionBase):
    """Configure the deterministic offline embedding model used by tests."""

    backend: Literal["fake"]
    dimension: int = Field(default=32, ge=2, le=16000)
    normalize: bool = True
    batch_size: int = Field(default=32, ge=1, le=512)
    embedding_space_id: str | None = Field(default=None, min_length=1)
    representation: Literal["dense", "sparse"] = "dense"
    similarity_metric: Literal["cosine", "dot", "euclidean"] = "cosine"


ModelDefinition = Annotated[
    OpenAICompatibleModelDefinition
    | SentenceTransformersModelDefinition
    | FakeModelDefinition,
    Field(discriminator="backend"),
]


class ComponentDefinition(BaseModel):
    """Describe one reusable component implementation and its own settings."""

    model_config = ConfigDict(
        extra="allow", frozen=True, str_strip_whitespace=True
    )

    id: str = Field(
        min_length=1,
        pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$",
        description="Stable internal component-configuration identifier.",
    )
    type: str = Field(min_length=1)
    model_id: str | None = Field(default=None, min_length=1)

    def settings(self) -> dict[str, object]:
        """Return operational fields for validation by the established schemas."""
        return self.model_dump(
            mode="json", exclude={"id", "type", "model_id"}
        )


class DeclarativeRepresentation(BaseModel):
    """Reference document-to-index components by internal IDs only."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    parser: str = Field(min_length=1)
    knowledge_builder: str | None = Field(default=None, min_length=1)
    chunker: str = Field(min_length=1)
    embedding: str | None = Field(default=None, min_length=1)
    vector_backend: str | None = Field(default=None, min_length=1)


class DeclarativeQuery(BaseModel):
    """Reference query-time components by internal IDs only."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    retriever: str = Field(min_length=1)
    reranker: str = Field(min_length=1)
    reasoner: str = Field(min_length=1)
    assessors: tuple[str, ...]
    decision_policy: str = Field(min_length=1)
    renderer: str = Field(min_length=1)
    chat_history: dict[str, object] | None = None


class DeclarativePipelineConfig(BaseModel):
    """Composition-only pipeline source resolved by the configuration registry."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    schema_version: Literal[2] = 2
    kind: Literal["ingestion", "query"]
    name: str = Field(min_length=1, max_length=255)
    version: int = Field(ge=1)
    description: str = Field(default="", max_length=4000)
    is_default: bool = False
    representation: DeclarativeRepresentation | None = None
    query: DeclarativeQuery | None = None

    @model_validator(mode="after")
    def validate_kind_sections(self) -> "DeclarativePipelineConfig":
        """Require only the composition sections belonging to the pipeline kind."""
        if self.kind == "ingestion" and self.representation is None:
            raise ValueError(f"{self.kind} pipelines require representation stages")
        if self.kind == "query" and self.query is None:
            raise ValueError(f"{self.kind} pipelines require query stages")
        if self.kind == "ingestion" and self.query is not None:
            raise ValueError("ingestion pipelines must not define query stages")
        if self.kind == "query" and self.representation is not None:
            raise ValueError("query pipelines must not define representation stages")
        return self
