"""HTTP schemas for the versioned evaluation protocol."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from llm_grounding.observability.models import RagRunObservation


class EvalQueryRequest(BaseModel):
    """Submit one grounded query against a corpus/project."""

    model_config = ConfigDict(extra="forbid")

    corpus_id: str = Field(min_length=1)
    query: str = Field(min_length=1, max_length=20_000)
    history: list[dict[str, Any]] = Field(default_factory=list)
    include: list[str] = Field(default_factory=list)
    filters: dict[str, Any] = Field(default_factory=dict)
    parameters: dict[str, Any] = Field(default_factory=dict)
    conversation_id: str | None = None
    turn_index: int = Field(default=0, ge=0)
    benchmark_case_id: str | None = None
    experiment_id: str | None = None
    seed: int | None = None
    perturbation_metadata: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    config_overrides: dict[str, Any] = Field(default_factory=dict)


class EvalRetrieveRequest(EvalQueryRequest):
    """Submit a retrieval-only request with the same corpus semantics."""


class EvalQueryResponse(BaseModel):
    """Return the canonical observation and protocol request identity."""

    observation: RagRunObservation
    request_id: str
    status: str


class EvalError(BaseModel):
    """Stable structured error envelope for evaluation clients."""

    error: Literal["eval_error"] = "eval_error"
    code: str
    message: str
    request_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class EvalHealthDependency(BaseModel):
    """Report one dependency state without exposing connection details."""

    status: Literal["healthy", "degraded", "unhealthy"]
    error: str | None = None


class EvalHealthTarget(BaseModel):
    """Identify the Target Protocol implementation being checked."""

    name: str
    version: str | None = None
    implementation: str | None = None


class EvalHealthDetails(BaseModel):
    """Expose dependency-level diagnostics for the target health check."""

    dependencies: dict[str, EvalHealthDependency]


class EvalHealthResponse(BaseModel):
    """Canonical Target Protocol v1 health response."""

    status: Literal["READY", "DEGRADED", "NOT_READY"]
    target: EvalHealthTarget
    details: EvalHealthDetails | None = None


class ProtocolHeaders(BaseModel):
    """Validated protocol metadata carried by evaluation requests."""

    request_id: str | None = None
    idempotency_key: str | None = None
    protocol: Literal["1"] = "1"


class EvalCorpusRequest(BaseModel):
    """Create or replace an isolated evaluation corpus inventory."""

    model_config = ConfigDict(extra="forbid")

    corpus_id: str | None = None
    parent_corpus_id: str | None = None
    documents: list[dict[str, Any]] = Field(default_factory=list)
    chunks: list[dict[str, Any]] = Field(default_factory=list)
    perturbation_metadata: dict[str, Any] = Field(default_factory=dict)
    version: int = Field(default=1, ge=1)


class EvalConfigOverrides(BaseModel):
    """Per-request settings supported by the current query pipeline."""

    model_config = ConfigDict(extra="forbid")

    parser: str | None = None
    chunker: str | None = None
    embedder: str | None = None
    embedding: dict[str, Any] | None = None
    vector_retriever: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=100)
    retrieval_top_k: int | None = Field(default=None, ge=1, le=100)
    rerank_top_k: int | None = Field(default=None, ge=1, le=100)
    reranker: str | None = None
    reranker_model_name: str | None = None
    reranker_device: str | None = None
    reranker_batch_size: int | None = Field(default=None, ge=1, le=512)
    context_chars: int | None = Field(default=None, ge=1, le=200_000)
    summary_per_facet: int | None = Field(default=None, ge=1, le=20)
    prompt_id: str | None = None
    model_name: str | None = None


class EvalInventoryUpload(BaseModel):
    """Upload isolated document or canonical chunk metadata."""

    items: list[dict[str, Any]] = Field(min_length=1, max_length=100_000)
