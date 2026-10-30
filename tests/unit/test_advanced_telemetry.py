"""Tests for optional local resource measurements."""

from llm_grounding.observability.telemetry import capture_process_usage
from llm_grounding.observability.trace import TraceRecorder


def test_process_telemetry_declares_local_scope() -> None:
    """Captured measurements never imply remote provider resource access."""
    usage = capture_process_usage("unit-test")

    assert usage.measurement_scope == "process"
    assert usage.measurement_source.startswith("stdlib")
    assert usage.metadata["provider_resource_usage"] == "unavailable"


def test_trace_span_attaches_resource_usage() -> None:
    """Completed spans carry scoped telemetry when capture is enabled."""
    recorder = TraceRecorder()
    span_id = recorder.start("component")
    recorder.end(span_id)

    assert recorder.spans[0].resource_usage is not None
    assert recorder.spans[0].resource_usage.span_id == span_id
