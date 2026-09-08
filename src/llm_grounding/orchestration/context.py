"""Runtime identity and append-only stage tracing, independent of semantic input."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from time import monotonic
from uuid import uuid4

from llm_grounding.domain.answers import JsonValue
from llm_grounding.domain.runs import TraceEvent


@dataclass
class RunContext:
    """Own one invocation's runtime identity, elapsed clock, and trace collection."""

    architecture_name: str
    architecture_version: int
    run_id: str = field(default_factory=lambda: str(uuid4()))
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, JsonValue] = field(default_factory=dict)
    trace: list[TraceEvent] = field(default_factory=list)
    _started_clock: float = field(default_factory=monotonic, repr=False)

    def record(self, stage: str, status: str) -> None:
        """Append a stage event without rewriting previous transitions."""
        self.trace.append(
            TraceEvent(
                stage, status, datetime.now(UTC), monotonic() - self._started_clock
            )
        )
