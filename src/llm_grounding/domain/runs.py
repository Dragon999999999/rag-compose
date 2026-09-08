"""Run records consumed by applications and offline evaluation.

Records retain intermediate results and sanitized failure information. A failed
run has no final answer; orchestration surfaces it on a RunFailedError rather
than silently treating a partial pipeline as successful.
"""

from dataclasses import dataclass, field
from datetime import datetime

from llm_grounding.domain.answers import (
    Answer,
    Assessment,
    DecisionResult,
    JsonValue,
    ReasoningResult,
)
from llm_grounding.domain.documents import DocumentSource
from llm_grounding.domain.evidence import RetrievalResult

DEFAULT_TASK = (
    "Extract keywords and a concise summary of the paper's main contribution."
)


@dataclass(frozen=True)
class RagRequest:
    """Supply source identity and the user's semantic task, without runtime state."""

    source: DocumentSource
    task: str = DEFAULT_TASK


@dataclass(frozen=True)
class TraceEvent:
    """Record a stage transition with UTC timestamp and elapsed run seconds."""

    stage: str
    status: str
    timestamp: datetime
    elapsed_seconds: float


@dataclass
class RagRun:
    """Retain one invocation, including partial outputs if a stage fails.

    Retrieval history permits repeated retrieval without mutating source evidence.
    The initial runner executes each stage once; this record does not require it.
    """

    run_id: str
    architecture_name: str
    architecture_version: int
    request: RagRequest
    started_at: datetime
    status: str = "running"
    retrieval_output: tuple[RetrievalResult, ...] = ()
    retrieval_history: list[tuple[RetrievalResult, ...]] = field(default_factory=list)
    reasoning_output: ReasoningResult | None = None
    assessments: tuple[Assessment, ...] = ()
    decision: DecisionResult | None = None
    answer: Answer | None = None
    trace: list[TraceEvent] = field(default_factory=list)
    metadata: dict[str, JsonValue] = field(default_factory=dict)
    completed_at: datetime | None = None
    error: str | None = None
