"""Strict, credential-free architecture configuration at the application boundary.

Unknown fields are rejected. Component names are checked against the registry
before any provider is constructed. Provider and model settings stay outside
domain models; only sanitized JSON configuration is retained in run metadata.
"""

from hashlib import sha256
from typing import Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    field_validator,
    model_validator,
)

from llm_grounding.ingestion.models import (
    ChunkerConfig,
    EmbeddingConfig,
    IngestionConfig,
)


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


class EmbedderConfig(ComponentConfig):
    """Configure an embedding provider used by retrieval-aware components.

    The nested model keeps provider connection settings consistent with the
    reasoning model configuration while the component type selects the
    embedding adapter. The component is optional for backward compatibility;
    stages such as full-document retrieval do not need to construct it.
    """

    type: str = Field(default="openai_compatible", min_length=1)
    model: ModelConfig
    batch_size: int = Field(default=32, ge=1, le=512)

    def identity(self) -> str:
        """Return a stable provider/model identity without endpoint details."""
        import json

        data = self.model_dump(mode="json", exclude={"batch_size"})
        data["model"].pop("base_url", None)
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return sha256(canonical.encode()).hexdigest()


class ReasonerConfig(ComponentConfig):
    """Select a reasoning implementation and its model configuration."""

    model: ModelConfig
    prompt_id: str | None = Field(
        default=None,
        min_length=1,
        description="Registered prompt ID selected for a prompt-driven reasoner.",
    )


class RetrieverConfig(ComponentConfig):
    """Select retrieval implementation and bounded context settings when applicable."""

    top_k: int | None = Field(default=None, ge=1, le=100)
    context_chars: int | None = Field(default=None, ge=1, le=200000)
    summary_per_facet: int | None = Field(default=None, ge=1, le=20)


class RerankerConfig(ComponentConfig):
    """Configure candidate truncation or model-based reranking.

    ``type`` is the established component name. ``provider`` is accepted as a
    clearer YAML/API spelling for new configurations and is normalized to
    ``type`` so existing persisted pipeline documents remain compatible.
    """

    type: str = Field(default="identity", min_length=1)
    provider: Literal["identity", "top_k", "cross_encoder"] | None = Field(
        default=None,
        exclude=True,
        description="Optional provider spelling for the reranker component type.",
    )
    retrieval_top_k: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description="Maximum candidates requested from the retriever before reranking.",
    )
    rerank_top_k: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description="Maximum candidates returned after reranking.",
    )
    model_name: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L6-v2",
        min_length=1,
        description="Sentence Transformers cross-encoder model identifier.",
    )
    device: str = Field(
        default="cpu",
        min_length=1,
        description="Inference device passed to the local cross-encoder.",
    )
    batch_size: int = Field(default=16, ge=1, le=512)

    @model_validator(mode="before")
    @classmethod
    def normalize_provider_input(cls, value: object) -> object:
        """Normalize the new provider spelling before Pydantic constructs the model."""
        if not isinstance(value, dict):
            return value
        provider = value.get("provider")
        if provider is None:
            return value
        selected = value.get("type", "identity")
        if selected not in ("identity", provider):
            raise ValueError("type and provider must select the same reranker")
        normalized = dict(value)
        normalized["type"] = provider
        return normalized

    @model_validator(mode="after")
    def validate_top_k(self) -> "RerankerConfig":
        """Validate top-k semantics after provider normalization."""
        if self.rerank_top_k is not None and self.retrieval_top_k is not None:
            if self.rerank_top_k > self.retrieval_top_k:
                raise ValueError("rerank_top_k must not exceed retrieval_top_k")
        if self.type == "identity" and self.rerank_top_k is not None:
            raise ValueError(
                "rerank_top_k requires the top_k or cross_encoder reranker"
            )
        return self


class ChatHistoryConfig(BaseModel):
    """Limit conversational context without treating it as corpus evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    mode: Literal["recent_messages"] = "recent_messages"
    max_messages: int = Field(default=10, ge=0, le=100)


class ArchitectureConfig(BaseModel):
    """Describe linear composition independently of the architecture's name."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    name: str = Field(min_length=1)
    version: int = Field(ge=0)
    parser: ComponentConfig
    knowledge_builder: ComponentConfig
    retriever: RetrieverConfig
    reranker: RerankerConfig = Field(
        validation_alias=AliasChoices("reranker", "reranking")
    )
    embedder: EmbedderConfig | None = None
    reasoner: ReasonerConfig
    assessors: tuple[ComponentConfig, ...]
    decision_policy: ComponentConfig
    renderer: ComponentConfig
    chat_history: ChatHistoryConfig | None = None

    def identity(self) -> str:
        """Hash architecture semantics without including endpoint credentials."""
        import json

        data = self.model_dump(mode="json")
        data["reasoner"]["model"].pop("base_url", None)
        if data.get("embedder") is not None:
            data["embedder"]["model"].pop("base_url", None)
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return sha256(canonical.encode()).hexdigest()

    @property
    def config_hash(self) -> str:
        """Return the deterministic architecture configuration hash."""
        return self.identity()


class VectorBackendConfig(ComponentConfig):
    """Select the registered vector backend and its immutable index contract."""

    type: Literal["qdrant"]
    index_version: str = Field(default="1", min_length=1, max_length=32)


class ModelReference(BaseModel):
    """Reference a credential-free entry in the shared model catalog."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    model_ref: str = Field(
        min_length=1,
        description="Stable model-catalog key, never a provider credential.",
    )


class PipelineReasonerConfig(ComponentConfig, ModelReference):
    """Select a reasoner and a named model-catalog entry."""

    prompt_id: str | None = Field(
        default=None,
        min_length=1,
        description="Registered prompt selected for a prompt-driven reasoner.",
    )


class PipelineEmbeddingConfig(ModelReference):
    """Select the named embedding model used to build and query an index."""


class PipelineRepresentationConfig(BaseModel):
    """Describe every document-to-index transformation in a pipeline."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    parser: ComponentConfig
    knowledge_builder: ComponentConfig
    chunker: ChunkerConfig
    embedding: PipelineEmbeddingConfig
    vector_backend: VectorBackendConfig

    def identity(self, catalog: "ModelCatalog | None" = None) -> str:
        """Hash all choices that can make a persisted vector index incompatible."""
        import json

        data = self.model_dump(mode="json")
        if catalog is not None:
            data["embedding_model"] = catalog.embedding(
                self.embedding.model_ref
            ).model_dump(mode="json")
        return sha256(
            json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


class PipelineQueryConfig(BaseModel):
    """Describe the query-time stages executed over a representation."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    retriever: RetrieverConfig
    reranker: RerankerConfig = Field(
        validation_alias=AliasChoices("reranker", "reranking")
    )
    reasoner: PipelineReasonerConfig
    assessors: tuple[ComponentConfig, ...]
    decision_policy: ComponentConfig
    renderer: ComponentConfig
    chat_history: ChatHistoryConfig | None = None


class ModelCatalog(BaseModel):
    """Credential-free named model definitions shared by pipeline YAML files."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    schema_version: Literal[1] = 1
    embedding_models: dict[str, EmbeddingConfig] = Field(default_factory=dict)
    reasoning_models: dict[str, ModelConfig] = Field(default_factory=dict)

    def embedding(self, model_ref: str) -> EmbeddingConfig:
        """Resolve an embedding reference or raise a configuration error."""
        try:
            return self.embedding_models[model_ref]
        except KeyError as exc:
            raise ValueError(f"Unknown embedding model reference: {model_ref}") from exc

    def reasoning(self, model_ref: str) -> ModelConfig:
        """Resolve a reasoning reference or raise a configuration error."""
        try:
            return self.reasoning_models[model_ref]
        except KeyError as exc:
            raise ValueError(f"Unknown reasoning model reference: {model_ref}") from exc


class PipelineConfig(BaseModel):
    """Versioned, validated full, ingestion, or query pipeline definition.

    Ingestion pipelines define indexed representations, query pipelines define
    chat-time behavior, and ``full`` retains the original combined contract.
    Pipeline YAML contains only component identifiers, tuning parameters, and
    model-catalog references. Provider credentials are resolved by infrastructure
    adapters and are never part of the persisted pipeline document.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    schema_version: Literal[1] = 1
    kind: Literal["full", "ingestion", "query"] = "full"
    name: str = Field(min_length=1, max_length=255)
    version: int = Field(ge=1)
    description: str = Field(default="", max_length=4000)
    is_default: bool = False
    representation: PipelineRepresentationConfig | None = None
    query: PipelineQueryConfig | None = None

    @model_validator(mode="after")
    def validate_kind_sections(self) -> "PipelineConfig":
        """Require exactly the stages that belong to the selected pipeline kind."""
        if self.kind in ("full", "ingestion") and self.representation is None:
            raise ValueError(f"{self.kind} pipelines require representation stages")
        if self.kind in ("full", "query") and self.query is None:
            raise ValueError(f"{self.kind} pipelines require query stages")
        if self.kind == "ingestion" and self.query is not None:
            raise ValueError("ingestion pipelines must not define query stages")
        if self.kind == "query" and self.representation is not None:
            raise ValueError("query pipelines must not define representation stages")
        return self

    def identity(self) -> str:
        """Return a stable hash of the pipeline's semantic configuration.

        ``kind`` was introduced after the original combined pipeline schema.
        A missing kind means ``full`` through Pydantic's default, so omitting
        the default from the canonical full-pipeline hash keeps records created
        before that schema migration semantically identical. Specialized
        ingestion and query pipelines retain their explicit kind in the hash.
        """
        import json

        data = self.model_dump(mode="json")
        if data.get("kind") == "full":
            data.pop("kind")
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return sha256(canonical.encode()).hexdigest()

    @property
    def config_hash(self) -> str:
        """Return the deterministic source configuration hash."""
        return self.identity()

    def representation_hash(self, catalog: ModelCatalog | None = None) -> str:
        """Return the identity used to decide whether an index can be reused."""
        if self.representation is None:
            return sha256(b"query-pipeline-no-representation").hexdigest()
        return self.representation.identity(catalog)

    def resolve(
        self, catalog: ModelCatalog
    ) -> tuple[ArchitectureConfig, IngestionConfig]:
        """Resolve named models into the legacy execution configurations.

        The execution engines still consume the established architecture and
        ingestion models. This adapter keeps their behavior stable while making
        pipeline configuration the persisted composition boundary.
        """
        if self.representation is None or self.query is None:
            raise ValueError("A full pipeline is required for combined resolution")
        embedding = catalog.embedding(self.representation.embedding.model_ref)
        reasoning = catalog.reasoning(self.query.reasoner.model_ref)
        architecture = ArchitectureConfig(
            name=self.name,
            version=self.version,
            parser=self.representation.parser,
            knowledge_builder=self.representation.knowledge_builder,
            retriever=self.query.retriever,
            reranker=self.query.reranker,
            embedder=None,
            reasoner=ReasonerConfig(
                type=self.query.reasoner.type,
                model=reasoning,
                prompt_id=self.query.reasoner.prompt_id,
            ),
            assessors=self.query.assessors,
            decision_policy=self.query.decision_policy,
            renderer=self.query.renderer,
            chat_history=self.query.chat_history,
        )
        ingestion = IngestionConfig.model_validate(
            {
                "parser": self.representation.parser.type,
                "knowledge_builder": self.representation.knowledge_builder.type,
                "chunker": self.representation.chunker.model_dump(mode="json"),
                "embedding": embedding.model_dump(mode="json"),
                "vector_backend": self.representation.vector_backend.type,
                "index_version": self.representation.vector_backend.index_version,
            }
        )
        return architecture, ingestion

    def resolve_ingestion(self, catalog: ModelCatalog) -> IngestionConfig:
        """Resolve an ingestion pipeline into the effective index configuration."""
        if self.representation is None:
            raise ValueError("An ingestion pipeline requires representation stages")
        embedding = catalog.embedding(self.representation.embedding.model_ref)
        return IngestionConfig.model_validate(
            {
                "parser": self.representation.parser.type,
                "knowledge_builder": self.representation.knowledge_builder.type,
                "chunker": self.representation.chunker.model_dump(mode="json"),
                "embedding": embedding.model_dump(mode="json"),
                "vector_backend": self.representation.vector_backend.type,
                "index_version": self.representation.vector_backend.index_version,
            }
        )

    def resolve_query(
        self, catalog: ModelCatalog, ingestion: IngestionConfig
    ) -> tuple[ArchitectureConfig, IngestionConfig]:
        """Resolve query stages against a project's already indexed representation."""
        if self.query is None:
            raise ValueError("A query pipeline requires query stages")
        reasoning = catalog.reasoning(self.query.reasoner.model_ref)
        architecture = ArchitectureConfig(
            name=self.name,
            version=self.version,
            parser=ComponentConfig(type=ingestion.parser),
            knowledge_builder=ComponentConfig(type=ingestion.knowledge_builder),
            retriever=self.query.retriever,
            reranker=self.query.reranker,
            embedder=None,
            reasoner=ReasonerConfig(
                type=self.query.reasoner.type,
                model=reasoning,
                prompt_id=self.query.reasoner.prompt_id,
            ),
            assessors=self.query.assessors,
            decision_policy=self.query.decision_policy,
            renderer=self.query.renderer,
            chat_history=self.query.chat_history,
        )
        return architecture, ingestion
