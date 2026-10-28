"""Typed standalone model, component, and pipeline configuration sources."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


def _integer_parameter(value: object, default: int | None) -> int | None:
    """Convert a dynamic YAML parameter to an integer when it is numeric."""
    if isinstance(value, (int, float, str)) and not isinstance(value, bool):
        return int(value)
    return default


class CredentialReference(BaseModel):
    """Name environment variables that provide a provider credential at runtime."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    env_var: str = Field(default="AQUEDUCT_API_KEY", min_length=1)
    file_env_var: str = Field(default="AQUEDUCT_API_KEY_FILE", min_length=1)


class ModelConnectionConfig(BaseModel):
    """Describe a remote model endpoint without storing credentials."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    endpoint: HttpUrl
    auth: CredentialReference = Field(default_factory=CredentialReference)
    timeout_seconds: float = Field(default=120, gt=0, le=3600, allow_inf_nan=False)
    retries: int = Field(default=2, ge=0, le=5)

    @model_validator(mode="after")
    def reject_url_credentials(self) -> "ModelConnectionConfig":
        """Keep credentials and query tokens out of endpoint configuration."""
        if (
            self.endpoint.username
            or self.endpoint.password
            or self.endpoint.query
            or self.endpoint.fragment
        ):
            raise ValueError(
                "connection.endpoint must not contain credentials, query, or fragment"
            )
        return self


class ModelSourceConfig(BaseModel):
    """Describe where local model weights can be obtained."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    type: Literal["huggingface", "url", "local_directory"]
    repository: str | None = Field(default=None, min_length=1)
    revision: str | None = Field(default=None, min_length=1)
    download_url: HttpUrl | None = None
    path: str | None = Field(default=None, min_length=1)
    sha256: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
        pattern=r"^[A-Fa-f0-9]{64}$",
    )

    @model_validator(mode="after")
    def validate_source(self) -> "ModelSourceConfig":
        """Require the source fields appropriate to the selected source type."""
        if self.type == "huggingface" and not self.repository:
            raise ValueError("Hugging Face sources require repository")
        if self.type == "url" and self.download_url is None:
            raise ValueError("URL model sources require download_url")
        if self.type == "local_directory" and not self.path:
            raise ValueError("Local-directory model sources require path")
        if self.type == "local_directory" and self.sha256 is not None:
            raise ValueError("Local-directory model sources do not support sha256")
        return self


class ModelStorageConfig(BaseModel):
    """Select a persistent path below the configured model-storage root."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    relative_path: str = Field(min_length=1)


class ModelRuntimeConfig(BaseModel):
    """Describe the optional inference runtime for a local model."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    type: Literal["docker", "in_process"] = "docker"
    image: str | None = Field(default=None, min_length=1)
    command: tuple[str, ...] = ()
    container_path: str = "/models/model"
    port: int = Field(default=8000, ge=1, le=65535)
    host_port: int | None = Field(default=None, ge=1, le=65535)
    readiness_path: str = "/health"
    readiness_timeout_seconds: float = Field(
        default=120, gt=0, le=3600, allow_inf_nan=False
    )
    idle_timeout_seconds: float = Field(
        default=600, ge=0, le=86400, allow_inf_nan=False
    )
    device: str = Field(default="cpu", min_length=1)
    environment: dict[str, str] = Field(default_factory=dict)
    cpu_limit: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    memory_limit: str | None = Field(default=None, min_length=1)
    gpu_count: int | None = Field(default=None, ge=1, le=128)

    @model_validator(mode="after")
    def validate_runtime(self) -> "ModelRuntimeConfig":
        """Require a Docker image for Docker-backed local models."""
        if self.type == "docker" and self.image is None:
            raise ValueError("Docker model runtimes require image")
        return self


class ModelDefinitionBase(BaseModel):
    """Common identity and provider fields for standalone model definitions."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    id: str = Field(
        min_length=1,
        pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$",
        description="Stable internal model-configuration identifier.",
    )
    stage: Literal["embedding", "reasoner", "reranker", "validation"] = Field(
        description="Pipeline stage allowed to reference this model."
    )
    name: str = Field(
        min_length=1,
        description="Provider or local runtime model name; not the internal ID.",
    )
    location: Literal["local", "remote"]
    provider: str | None = Field(default=None, min_length=1)
    connection: ModelConnectionConfig | None = None
    source: ModelSourceConfig | None = None
    storage: ModelStorageConfig | None = None
    runtime: ModelRuntimeConfig | None = None
    parameters: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_shape(cls, value: object) -> object:
        """Accept the former flat shape while canonical YAML uses nested fields."""
        if not isinstance(value, dict) or "location" in value:
            return value
        normalized = dict(value)
        backend = normalized.get("backend")
        location = "remote" if backend == "openai_compatible" else "local"
        normalized["location"] = location
        parameter_names = {
            "dimension",
            "dimensions",
            "normalize",
            "normalization",
            "batch_size",
            "embedding_space_id",
            "representation",
            "similarity_metric",
        }
        parameters = dict(normalized.get("parameters", {}))
        for name in parameter_names:
            if name in normalized:
                parameters[name] = normalized.pop(name)
        normalized["parameters"] = parameters
        if "base_url" in normalized:
            normalized["connection"] = {
                "endpoint": normalized.pop("base_url"),
                "auth": normalized.pop("auth", {}),
                "timeout_seconds": normalized.pop("timeout_seconds", 120),
                "retries": normalized.pop("retries", 2),
            }
        if location == "local":
            normalized.setdefault(
                "source",
                {"type": "huggingface", "repository": normalized["name"]},
            )
            normalized.setdefault(
                "storage",
                {"relative_path": f"{normalized['stage']}/{normalized['id']}"},
            )
            normalized.setdefault(
                "runtime", {"type": "in_process", "idle_timeout_seconds": 0}
            )
        return normalized

    @property
    def base_url(self) -> HttpUrl | None:
        """Return the legacy endpoint spelling for existing provider adapters."""
        return self.connection.endpoint if self.connection is not None else None

    @property
    def auth(self) -> CredentialReference:
        """Return remote credential references without exposing credential values."""
        return (
            self.connection.auth
            if self.connection is not None
            else CredentialReference()
        )

    @property
    def timeout_seconds(self) -> float:
        """Return the configured remote request deadline."""
        return self.connection.timeout_seconds if self.connection is not None else 120

    @property
    def retries(self) -> int:
        """Return the configured remote retry count."""
        return self.connection.retries if self.connection is not None else 0

    def parameter(self, name: str, default: object = None) -> object:
        """Read one backend-specific model parameter."""
        return self.parameters.get(name, default)


class OpenAICompatibleModelDefinition(ModelDefinitionBase):
    """Configure a model served through an OpenAI-compatible HTTP endpoint."""

    backend: Literal["openai_compatible"]
    provider: Literal["aqueduct"] = "aqueduct"
    location: Literal["remote"] = "remote"

    @model_validator(mode="after")
    def require_connection(self) -> "OpenAICompatibleModelDefinition":
        """Require endpoint settings for remote OpenAI-compatible models."""
        if self.connection is None:
            raise ValueError("Remote OpenAI-compatible models require connection")
        return self

    @property
    def dimension(self) -> int | None:
        """Return the optional embedding dimension."""
        value = self.parameter("dimension", self.parameter("dimensions"))
        return _integer_parameter(value, None)

    @property
    def embedding_space_id(self) -> str | None:
        """Return the optional provider-independent embedding-space ID."""
        value = self.parameter("embedding_space_id")
        return value if isinstance(value, str) else None

    @property
    def representation(self) -> Literal["dense", "sparse", "multivector"]:
        """Return the configured vector representation."""
        value = self.parameter("representation", "dense")
        return value if value in {"dense", "sparse", "multivector"} else "dense"

    @property
    def similarity_metric(self) -> Literal["cosine", "dot", "euclidean"]:
        """Return the configured vector similarity metric."""
        value = self.parameter("similarity_metric", "cosine")
        return value if value in {"cosine", "dot", "euclidean"} else "cosine"


class SentenceTransformersModelDefinition(ModelDefinitionBase):
    """Configure a locally loaded Transformer or GLiNER model."""

    backend: Literal["sentence_transformers", "gliner"]
    location: Literal["local"] = "local"

    @property
    def revision(self) -> str | None:
        """Return the optional pinned source revision."""
        return self.source.revision if self.source is not None else None

    @property
    def dimension(self) -> int:
        """Return the configured embedding dimension."""
        return (
            _integer_parameter(
                self.parameter("dimension", self.parameter("dimensions", 384)), 384
            )
            or 384
        )

    @property
    def normalize(self) -> bool:
        """Return whether provider vectors should be normalized."""
        return bool(self.parameter("normalize", self.parameter("normalization", True)))

    @property
    def batch_size(self) -> int:
        """Return the configured local inference batch size."""
        return _integer_parameter(self.parameter("batch_size", 32), 32) or 32

    @property
    def cache_dir_env(self) -> str | None:
        """Return the optional legacy cache-directory environment variable."""
        value = self.parameter("cache_dir_env")
        return value if isinstance(value, str) else None

    @property
    def query_prefix(self) -> str:
        """Return text prepended to query inputs before local inference."""
        value = self.parameter("query_prefix", "")
        return value if isinstance(value, str) else ""

    @property
    def document_prefix(self) -> str:
        """Return text prepended to document inputs before local inference."""
        value = self.parameter("document_prefix", "")
        return value if isinstance(value, str) else ""

    @property
    def trust_remote_code(self) -> bool:
        """Return whether the local runtime must load model-provided code."""
        return bool(self.parameter("trust_remote_code", False))

    @property
    def embedding_space_id(self) -> str | None:
        """Return the optional provider-independent embedding-space ID."""
        value = self.parameter("embedding_space_id")
        return value if isinstance(value, str) else None

    @property
    def representation(self) -> Literal["dense", "sparse", "multivector"]:
        """Return the configured vector representation."""
        value = self.parameter("representation", "dense")
        return value if value in {"dense", "sparse", "multivector"} else "dense"

    @property
    def similarity_metric(self) -> Literal["cosine", "dot", "euclidean"]:
        """Return the configured vector similarity metric."""
        value = self.parameter("similarity_metric", "cosine")
        return value if value in {"cosine", "dot", "euclidean"} else "cosine"


class FakeModelDefinition(ModelDefinitionBase):
    """Configure the deterministic offline embedding model used by tests."""

    backend: Literal["fake"]
    location: Literal["local"] = "local"

    @property
    def dimension(self) -> int:
        """Return the configured deterministic vector dimension."""
        return (
            _integer_parameter(
                self.parameter("dimension", self.parameter("dimensions", 32)), 32
            )
            or 32
        )

    @property
    def normalize(self) -> bool:
        """Return whether fake vectors are normalized."""
        return bool(self.parameter("normalize", True))

    @property
    def batch_size(self) -> int:
        """Return the configured deterministic batch size."""
        return _integer_parameter(self.parameter("batch_size", 32), 32) or 32

    @property
    def embedding_space_id(self) -> str | None:
        """Return the optional provider-independent embedding-space ID."""
        value = self.parameter("embedding_space_id")
        return value if isinstance(value, str) else None

    @property
    def representation(self) -> Literal["dense", "sparse"]:
        """Return the configured vector representation."""
        value = self.parameter("representation", "dense")
        return value if value in {"dense", "sparse"} else "dense"

    @property
    def similarity_metric(self) -> Literal["cosine", "dot", "euclidean"]:
        """Return the configured vector similarity metric."""
        value = self.parameter("similarity_metric", "cosine")
        return value if value in {"cosine", "dot", "euclidean"} else "cosine"


ModelDefinition = Annotated[
    OpenAICompatibleModelDefinition
    | SentenceTransformersModelDefinition
    | FakeModelDefinition,
    Field(discriminator="backend"),
]


class ComponentRequirements(BaseModel):
    """Declare resources that must exist before a component can execute."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    required: tuple[str, ...] = ()


class ComponentDefinition(BaseModel):
    """Describe one reusable component with a category and implementation."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    id: str = Field(
        min_length=1,
        pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$",
        description="Stable internal component-configuration identifier.",
    )
    type: str = Field(min_length=1)
    implementation: str = Field(min_length=1)
    description: str = Field(default="", max_length=4000)
    config: dict[str, object] = Field(default_factory=dict)
    requirements: ComponentRequirements | None = None
    input_terms: tuple[str, ...] = ()
    output_terms: tuple[str, ...] = ()

    def settings(self) -> dict[str, object]:
        """Return nested implementation settings for typed runtime validation."""
        return dict(self.config)

    @property
    def model_id(self) -> str | None:
        """Return a model reference stored in the implementation config."""
        value = self.config.get("model_id")
        return value if isinstance(value, str) else None


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
    execution: dict[str, object] | None = None


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
    stage_overrides: dict[str, dict[str, object]] = Field(default_factory=dict)

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
