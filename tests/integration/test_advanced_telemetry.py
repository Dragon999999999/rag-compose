"""Acceptance checks for scoped optional resource telemetry."""

from llm_grounding.observability.telemetry import capture_process_usage


def test_telemetry_scope_is_explicit_and_provider_usage_is_not_claimed() -> None:
    """Local measurements identify their scope and leave remote data unavailable."""
    usage = capture_process_usage("integration")

    assert usage.measurement_scope == "process"
    assert usage.metadata["provider_resource_usage"] == "unavailable"
