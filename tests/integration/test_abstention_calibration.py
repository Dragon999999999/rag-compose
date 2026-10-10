"""Acceptance checks for explicit abstention and confidence semantics."""

from llm_grounding.observability.models import Confidence, ObservationAnswer


def test_abstention_is_not_failure_and_confidence_can_be_unavailable() -> None:
    """Abstention remains a completed policy outcome with no invented confidence."""
    answer = ObservationAnswer(
        status="abstained",
        generation_status="completed",
        insufficient_evidence=True,
        confidence=Confidence(),
    )

    assert answer.status == "abstained"
    assert answer.status != "failed"
    assert answer.confidence.value is None
