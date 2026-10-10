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


class ProtocolHeaders(BaseModel):
    """Validated protocol metadata carried by evaluation requests."""

    request_id: str | None = None
    idempotency_key: str | None = None
    protocol: Literal["1"] = "1"
