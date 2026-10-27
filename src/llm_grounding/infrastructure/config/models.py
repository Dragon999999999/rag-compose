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
    """Select a component category and its registered implementation."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    id: str | None = Field(default=None, min_length=1)
    type: str = Field(min_length=1, description="General component category.")
    implementation: str | None = Field(
        default=None, min_length=1, description="Registered runtime implementation."
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_component_identity(cls, value: object) -> object:
        """Normalize old direct implementation inputs for in-memory callers."""
        if not isinstance(value, dict) or value.get("implementation") is not None:
            return value
        aliases = {
            "pymupdf": ("parser", "pymupdf"),
            "block_aware": ("chunker", "block_aware"),
            "fixed_window": ("chunker", "fixed_window"),
            "recursive_token": ("chunker", "recursive_token"),
            "hierarchical": ("chunker", "hierarchical"),
            "structural": ("chunker", "structural"),
            "sentence_window": ("chunker", "sentence_window"),
            "basic": ("knowledge_builder", "basic"),
            "vector": ("dense", "vector_similarity"),
            "postgres_fts": ("sparse", "postgres_fts"),
            "bm25": ("sparse", "bm25"),
            "splade": ("sparse", "splade"),
            "splade_v3": ("sparse", "splade_v3"),
            "bge_m3_sparse": ("sparse", "bge_m3_sparse"),
            "colbert": ("multivector", "colbert_v2"),
            "colbert_v2": ("multivector", "colbert_v2"),
            "hybrid_rrf": ("hybrid", "hybrid_rrf"),
            "hybrid_weighted": ("hybrid", "hybrid_weighted"),
            "full_context": ("full_context", "postgres_all_chunks"),
            "metadata_rules": ("knowledge_builder", "metadata_rules"),
            "metadata_spacy": ("knowledge_builder", "metadata_spacy"),
            "metadata_gliner": ("knowledge_builder", "metadata_gliner"),
            "identity": ("reranker", "identity"),
            "top_k": ("reranker", "top_k"),
            "cross_encoder": ("reranker", "cross_encoder"),
            "generative": ("reranker", "generative"),
            "basic_grounded": ("reasoner", "basic_grounded"),
            "source_reference": ("assessor", "source_reference"),
            "source_references": ("assessor", "source_reference"),
            "noop": ("assessor", "noop"),
            "retrieval_confidence": ("assessor", "retrieval_confidence"),
            "relevance": ("assessor", "relevance"),
            "evidence_sufficiency": ("assessor", "evidence_sufficiency"),
            "diversity": ("assessor", "diversity"),
            "basic_grounding": ("decision_policy", "basic_grounding"),
            "grounded_answer": ("renderer", "grounded_answer"),
            "qdrant": ("vector_backend", "qdrant"),
            "pgvector": ("vector_backend", "pgvector"),
            "faiss": ("vector_backend", "faiss"),
        }
        selected_type = value.get("type")
        if not isinstance(selected_type, str):
            return value
        selected = aliases.get(selected_type)
        if selected is None:
            return value
        normalized = dict(value)
        normalized["type"], normalized["implementation"] = selected
        return normalized


class AssessorConfig(ComponentConfig):
    """Configure one deterministic assessor and its measurement thresholds."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    type: str = Field(default="assessor", min_length=1)
    implementation: Literal[
        "noop",
        "source_reference",
        "retrieval_confidence",
        "relevance",
        "evidence_sufficiency",
        "diversity",
    ] = "source_reference"
    threshold: float | None = Field(default=None, allow_inf_nan=False)
    score_kind: str | None = Field(default=None, min_length=1)
    min_threshold_count: int = Field(default=0, ge=0, le=10000)
    fallback_to_retrieval: bool = False
    requested_aspects: tuple[str, ...] = ()
    max_pairs: int = Field(default=1000, ge=1, le=100000)


class ModelConfig(BaseModel):
    """Configure one resolved OpenAI-compatible model connection."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    provider: Literal["aqueduct"]
    name: str = Field(
        min_length=1, description="Model identifier served by the endpoint."
    )
    base_url: HttpUrl
    api_key_env: str = Field(default="AQUEDUCT_API_KEY", min_length=1)
    api_key_file_env: str = Field(default="AQUEDUCT_API_KEY_FILE", min_length=1)
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
    prompt_id: str | None = Field(
        default=None,
        min_length=1,
        description="Registered prompt ID selected for a prompt-driven reasoner.",
    )


class KnowledgeBuilderConfig(ComponentConfig):
    """Configure canonical structural or metadata knowledge construction."""

    type: str = Field(default="knowledge_builder", min_length=1)
    implementation: Literal[
        "basic", "structural", "metadata_rules", "metadata_spacy", "metadata_gliner"
    ] = "basic"
    model_id: str | None = Field(
        default=None,
        min_length=1,
        description="Registered local model used by a model-backed builder.",
    )
    rule_patterns: dict[str, str] = Field(default_factory=dict)
    max_keywords: int = Field(default=12, ge=0, le=100)
    keyword_min_length: int = Field(default=4, ge=1, le=100)
    stopwords: tuple[str, ...] = ()
    spacy_model: str = Field(default="en_core_web_sm", min_length=1)
    spacy_entity_labels: tuple[str, ...] = ()
    include_linguistic_metadata: bool = False
    entity_labels: tuple[str, ...] = (
        "person",
        "organization",
        "location",
        "date",
        "product",
    )
    confidence_threshold: float = Field(default=0.35, ge=0, le=1)
    batch_size: int = Field(default=4, ge=1, le=128)


RetrieverRequirement = Literal[
    "persisted_chunks",
    "postgres_fts",
    "dense_vector_index",
    "compatible_query_embedder",
    "sparse_vector_index",
    "multivector_index",
    "sparse_query_embedder",
    "late_interaction_model",
]


class RetrieverRequirements(BaseModel):
    """Resources a retriever must have before it can execute."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    required: tuple[RetrieverRequirement, ...] = (
        "dense_vector_index",
        "compatible_query_embedder",
    )

    @model_validator(mode="after")
    def validate_unique_requirements(self) -> "RetrieverRequirements":
        """Reject duplicate resource declarations in component YAML."""
        if len(set(self.required)) != len(self.required):
            raise ValueError("Retriever requirements must be unique")
        return self


class RetrieverConfig(ComponentConfig):
    """Select retrieval implementation and bounded context settings when applicable."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    top_k: int | None = Field(default=None, ge=1, le=100)
    context_chars: int | None = Field(default=None, ge=1, le=200000)
    context_budget_tokens: int | None = Field(default=None, ge=1, le=1_000_000)
    summary_per_facet: int | None = Field(default=None, ge=1, le=20)
    dense_top_k: int | None = Field(default=None, ge=1, le=100)
    sparse_top_k: int | None = Field(default=None, ge=1, le=100)
    rrf_k: int = Field(default=60, ge=1, le=10_000)
    dense_weight: float = Field(default=1.0, gt=0, le=100, allow_inf_nan=False)
    sparse_weight: float = Field(default=1.0, gt=0, le=100, allow_inf_nan=False)
    bm25_k1: float = Field(default=1.2, gt=0, le=10, allow_inf_nan=False)
    bm25_b: float = Field(default=0.75, ge=0, le=1, allow_inf_nan=False)
    candidate_top_k: int | None = Field(default=None, ge=1, le=200)
    score_normalization: Literal["min_max", "none"] = "min_max"
    query_embedder: str | None = Field(
        default=None,
        min_length=1,
        description="Optional embedding model-catalog reference for query encoding.",
    )
    requirements: RetrieverRequirements = Field(
        default_factory=RetrieverRequirements
    )


class RerankerConfig(ComponentConfig):
    """Configure candidate truncation or model-based reranking.

    ``type`` identifies the reranker family and ``implementation`` selects the
    concrete algorithm. ``provider`` remains an input alias for API callers.
    """

    type: str = Field(default="reranker", min_length=1)
    implementation: str = Field(default="identity", min_length=1)
    provider: Literal[
        "identity", "top_k", "cross_encoder", "generative"
    ] | None = Field(
        default=None,
        exclude=True,
        description="Optional provider spelling for the reranker component type.",
    )
    model_id: str | None = Field(
        default=None,
        min_length=1,
        description="Registered local model configuration used for reranking.",
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
        selected = value.get("implementation", value.get("type", "identity"))
        if selected not in ("identity", provider, "reranker"):
            raise ValueError(
                "implementation and provider must select the same reranker"
            )
        normalized = dict(value)
        normalized["type"] = provider
        normalized["implementation"] = provider
        return normalized

    @model_validator(mode="after")
    def validate_top_k(self) -> "RerankerConfig":
        """Validate top-k semantics after provider normalization."""
        if self.rerank_top_k is not None and self.retrieval_top_k is not None:
            if self.rerank_top_k > self.retrieval_top_k:
                raise ValueError("rerank_top_k must not exceed retrieval_top_k")
        if self.implementation == "identity" and self.rerank_top_k is not None:
            raise ValueError(
                "rerank_top_k requires the top_k or model-based reranker"
            )
        return self


class ChatHistoryConfig(BaseModel):
    """Limit conversational context without treating it as corpus evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    mode: Literal["recent_messages"] = "recent_messages"
    max_messages: int = Field(default=10, ge=0, le=100)


class ArchitectureConfig(BaseModel):
    """Describe the effective query composition used by the runtime."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    name: str = Field(min_length=1)
    version: int = Field(ge=0)
    parser: ComponentConfig
    knowledge_builder: ComponentConfig | None = None
    retriever: RetrieverConfig
    reranker: RerankerConfig = Field(
        validation_alias=AliasChoices("reranker", "reranking")
    )
    reasoner: ReasonerConfig
    assessors: tuple[AssessorConfig, ...]
    decision_policy: ComponentConfig
    renderer: ComponentConfig
    chat_history: ChatHistoryConfig | None = None

    def identity(self) -> str:
        """Hash architecture semantics without including endpoint credentials."""
        import json

        data = self.model_dump(mode="json")
        data["reasoner"]["model"].pop("base_url", None)
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return sha256(canonical.encode()).hexdigest()

    @property
    def config_hash(self) -> str:
        """Return the deterministic architecture configuration hash."""
        return self.identity()


class VectorBackendConfig(ComponentConfig):
    """Select the registered vector backend and its immutable index contract."""

    type: str = Field(default="vector_backend", min_length=1)
    implementation: str = Field(default="qdrant", min_length=1)
    index_version: str = Field(default="1", min_length=1, max_length=32)
    representation: Literal["dense"] = "dense"
    similarity_metric: Literal["cosine", "dot", "euclidean"] = "cosine"
    index_type: Literal["flat", "hnsw", "ivf"] = "flat"
    hnsw_m: int = Field(default=16, ge=4, le=256)
    hnsw_ef_construction: int = Field(default=100, ge=4, le=10000)
    hnsw_ef_search: int = Field(default=64, ge=1, le=10000)
    nlist: int = Field(default=100, ge=1, le=100000)
    nprobe: int = Field(default=8, ge=1, le=100000)
    idle_timeout_seconds: float = Field(default=600, ge=0, le=86400)
    storage_path: str | None = Field(default=None, min_length=1)

    @model_validator(mode="before")
    @classmethod
    def normalize_backend_input(cls, value: object) -> object:
        """Treat a direct backend identifier as its implementation selection."""
        if not isinstance(value, dict):
            return value
        selected = value.get("type")
        if selected not in {None, "vector_backend"} and "implementation" not in value:
            normalized = dict(value)
            normalized["type"] = "vector_backend"
            normalized["implementation"] = selected
            return normalized
        return value


class ApplicationDefaults(BaseModel):
    """Select application-wide defaults without hardcoding backend identifiers."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    schema_version: Literal[1] = 1
    vector_backend: str | None = Field(default=None, min_length=1)


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

    component_id: str | None = Field(default=None, min_length=1)


class PipelineRepresentationConfig(BaseModel):
    """Describe every document-to-index transformation in a pipeline."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    parser: ComponentConfig
    knowledge_builder: KnowledgeBuilderConfig | None = None
    chunker: ChunkerConfig
    embedding: PipelineEmbeddingConfig | None = None
    vector_backend: VectorBackendConfig | None = None

    def identity(self, catalog: "ModelCatalog | None" = None) -> str:
        """Hash all choices that can make a persisted vector index incompatible."""
        import json

        data = self.model_dump(mode="json")
        if self.knowledge_builder is not None:
            data["knowledge_builder"] = self.knowledge_builder.model_dump(
                mode="json", exclude_unset=True
            )
        backend = data.get("vector_backend")
        if isinstance(backend, dict):
            backend.pop("representation", None)
            backend.pop("similarity_metric", None)
        if catalog is not None and self.embedding is not None:
            embedding_config = catalog.embedding(self.embedding.model_ref)
            data["embedding_model"] = (
                embedding_config.specification().model_dump(mode="json")
                if embedding_config.embedding_space_id is not None
                else embedding_config.model_dump(mode="json")
            )
            if embedding_config.embedding_space_id is None:
                if data["embedding_model"].get("embedding_space_id") is None:
                    data["embedding_model"].pop("embedding_space_id", None)
                if data["embedding_model"].get("representation") == "dense":
                    data["embedding_model"].pop("representation", None)
                if data["embedding_model"].get("similarity_metric") == "cosine":
                    data["embedding_model"].pop("similarity_metric", None)
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
    assessors: tuple[AssessorConfig, ...]
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
    """Versioned, validated ingestion or query pipeline definition.

    Pipeline YAML contains only component identifiers, tuning parameters, and
    model-catalog references. Provider credentials are resolved by infrastructure
    adapters and are never part of the persisted pipeline document.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    schema_version: Literal[1] = 1
    kind: Literal["ingestion", "query"]
    name: str = Field(min_length=1, max_length=255)
    version: int = Field(ge=1)
    description: str = Field(default="", max_length=4000)
    is_default: bool = False
    representation: PipelineRepresentationConfig | None = None
    query: PipelineQueryConfig | None = None
    stage_overrides: dict[str, dict[str, object]] = Field(default_factory=dict)
    resolved_catalog: ModelCatalog | None = Field(
        default=None,
        description=(
            "Immutable provider configuration resolved when the pipeline was "
            "persisted."
        ),
    )

    @model_validator(mode="after")
    def validate_kind_sections(self) -> "PipelineConfig":
        """Require exactly the stages that belong to the selected pipeline kind."""
        if self.kind == "ingestion" and self.representation is None:
            raise ValueError(f"{self.kind} pipelines require representation stages")
        if self.kind == "query" and self.query is None:
            raise ValueError(f"{self.kind} pipelines require query stages")
        if self.kind == "ingestion" and self.query is not None:
            raise ValueError("ingestion pipelines must not define query stages")
        if self.kind == "query" and self.representation is not None:
            raise ValueError("query pipelines must not define representation stages")
        if self.kind == "ingestion" and self.stage_overrides:
            raise ValueError("ingestion pipelines must not define stage overrides")
        return self

    def identity(self) -> str:
        """Return a stable hash of the pipeline's semantic configuration."""
        import json

        data = self.model_dump(mode="json")
        representation = data.get("representation")
        if isinstance(representation, dict):
            if (
                self.representation is not None
                and self.representation.knowledge_builder
            ):
                representation["knowledge_builder"] = (
                    self.representation.knowledge_builder.model_dump(
                        mode="json", exclude_unset=True
                    )
                )
            backend = representation.get("vector_backend")
            if isinstance(backend, dict):
                backend.pop("representation", None)
                backend.pop("similarity_metric", None)
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
        catalog = self.effective_catalog(catalog)
        return self.representation.identity(catalog)

    def with_resolved_catalog(self, catalog: ModelCatalog) -> "PipelineConfig":
        """Attach only the referenced model definitions to a persisted snapshot."""
        embedding_refs: set[str] = set()
        reasoning_refs: set[str] = set()
        if (
            self.representation is not None
            and self.representation.embedding is not None
        ):
            embedding_refs.add(self.representation.embedding.model_ref)
        if self.query is not None:
            if self.query.retriever.query_embedder is not None:
                embedding_refs.add(self.query.retriever.query_embedder)
            retriever_overrides = self.stage_overrides.get("retriever", {})
            query_embedder = retriever_overrides.get("query_embedder")
            if isinstance(query_embedder, str):
                embedding_refs.add(query_embedder)
            reasoning_refs.add(self.query.reasoner.model_ref)
        return self.model_copy(
            update={
                "resolved_catalog": ModelCatalog(
                    embedding_models={
                        ref: catalog.embedding(ref) for ref in sorted(embedding_refs)
                    },
                    reasoning_models={
                        ref: catalog.reasoning(ref) for ref in sorted(reasoning_refs)
                    },
                )
            }
        )

    def effective_catalog(self, fallback: ModelCatalog | None = None) -> ModelCatalog:
        """Return the persisted model snapshot, or the supplied current catalog."""
        if self.resolved_catalog is not None:
            return self.resolved_catalog
        if fallback is None:
            raise ValueError(
                "A model catalog is required for unresolved pipeline config"
            )
        return fallback

    def resolve_ingestion(self, catalog: ModelCatalog) -> IngestionConfig:
        """Resolve an ingestion pipeline into the effective index configuration."""
        catalog = self.effective_catalog(catalog)
        if self.representation is None:
            raise ValueError("An ingestion pipeline requires representation stages")
        embedding = (
            catalog.embedding(self.representation.embedding.model_ref)
            if self.representation.embedding is not None
            else None
        )
        return IngestionConfig.model_validate(
            {
                "parser": self.representation.parser.implementation,
                "knowledge_builder": (
                    self.representation.knowledge_builder.implementation
                    if self.representation.knowledge_builder is not None
                    else None
                ),
                "knowledge_builder_config": (
                    self.representation.knowledge_builder.model_dump(
                        mode="json",
                        exclude={"id", "type", "implementation"},
                        exclude_unset=True,
                    )
                    if self.representation.knowledge_builder is not None
                    else {}
                ),
                "chunker": self.representation.chunker.model_dump(mode="json"),
                "embedding": embedding.model_dump(mode="json") if embedding else None,
                "vector_backend": (
                    self.representation.vector_backend.implementation
                    if self.representation.vector_backend is not None
                    else None
                ),
                "vector_backend_config": (
                    self.representation.vector_backend.model_dump(mode="json")
                    if self.representation.vector_backend is not None
                    else {}
                ),
                "index_version": (
                    self.representation.vector_backend.index_version
                    if self.representation.vector_backend is not None
                    else "1"
                ),
            }
        )

    def resolve_query(
        self, catalog: ModelCatalog, ingestion: IngestionConfig
    ) -> tuple[ArchitectureConfig, IngestionConfig]:
        """Resolve query stages against a project's already indexed representation."""
        catalog = self.effective_catalog(catalog)
        if self.query is None:
            raise ValueError("A query pipeline requires query stages")
        reasoning = catalog.reasoning(self.query.reasoner.model_ref)
        architecture = ArchitectureConfig(
            name=self.name,
            version=self.version,
            parser=ComponentConfig(type="parser", implementation=ingestion.parser),
            knowledge_builder=(
                ComponentConfig(
                    type="knowledge_builder",
                    implementation=ingestion.knowledge_builder,
                )
                if ingestion.knowledge_builder is not None
                else None
            ),
            retriever=self.query.retriever,
            reranker=self.query.reranker,
            reasoner=ReasonerConfig(
                type="reasoner",
                implementation=self.query.reasoner.implementation,
                model=reasoning,
                prompt_id=self.query.reasoner.prompt_id,
            ),
            assessors=self.query.assessors,
            decision_policy=self.query.decision_policy,
            renderer=self.query.renderer,
            chat_history=self.query.chat_history,
        )
        return architecture, ingestion
