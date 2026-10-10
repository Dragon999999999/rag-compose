"""Lightweight nested tracing that retains failed spans."""

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from time import monotonic
from uuid import uuid4

from llm_grounding.observability.models import StageType, TraceSpan
from llm_grounding.observability.telemetry import (
    ProcessSample,
    capture_process_usage,
    process_sample,
)


class TraceRecorder:
    """Record nested spans with wall-clock timestamps and monotonic durations."""

    def __init__(self, *, capture_resources: bool = True) -> None:
        """Create an empty recorder, optionally attaching local process usage."""
        self._spans: list[TraceSpan] = []
        self._active: list[tuple[str, float, ProcessSample | None]] = []
        self._capture_resources = capture_resources

    @property
    def spans(self) -> tuple[TraceSpan, ...]:
        """Return spans in completion-independent creation order."""
        return tuple(self._spans)

    def start(
        self,
        name: str,
        *,
        stage_type: StageType | None = None,
        input_count: int | None = None,
        metadata: dict[str, object] | None = None,
    ) -> str:
        """Start a span and return its identifier."""
        span_id = str(uuid4())
        self._spans.append(
            TraceSpan(
                span_id=span_id,
                name=name,
                stage_type=stage_type,
                parent_span_id=self._active[-1][0] if self._active else None,
                started_at=datetime.now(UTC),
                input_count=input_count,
                metadata=metadata or {},
            )
        )
        self._active.append(
            (
                span_id,
                monotonic(),
                process_sample() if self._capture_resources else None,
            )
        )
        return span_id

    def end(
        self,
        span_id: str,
        *,
        status: str = "completed",
        output_count: int | None = None,
        error: str | None = None,
    ) -> TraceSpan:
        """Complete a span while preserving a sanitized failure status."""
        index = next(
            (
                index
                for index, span in enumerate(self._spans)
                if span.span_id == span_id
            ),
            None,
        )
        if index is None:
            raise KeyError(f"Unknown trace span: {span_id}")
        if self._active and self._active[-1][0] == span_id:
            _, started_monotonic, started_process = self._active.pop()
        else:
            active_index = next(
                index for index, item in enumerate(self._active) if item[0] == span_id
            )
            _, started_monotonic, started_process = self._active.pop(active_index)
        span = self._spans[index]
        updated = span.model_copy(
            update={
                "ended_at": datetime.now(UTC),
                "duration_ms": max(0.0, (monotonic() - started_monotonic) * 1000),
                "status": status,
                "output_count": output_count,
                "error": error,
                "resource_usage": (
                    capture_process_usage(
                        span.name, span_id=span_id, started=started_process
                    )
                    if self._capture_resources
                    else None
                ),
            }
        )
        self._spans[index] = updated
        return updated

    @contextmanager
    def span(
        self,
        name: str,
        *,
        stage_type: StageType | None = None,
        input_count: int | None = None,
        metadata: dict[str, object] | None = None,
    ) -> Iterator[str]:
        """Record a span around a synchronous block and retain failures."""
        span_id = self.start(
            name,
            stage_type=stage_type,
            input_count=input_count,
            metadata=metadata,
        )
        try:
            yield span_id
        except Exception as exc:
            self.end(span_id, status="failed", error=type(exc).__name__)
            raise
        else:
            self.end(span_id)
