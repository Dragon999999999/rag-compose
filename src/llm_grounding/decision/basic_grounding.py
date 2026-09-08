"""Apply basic source-ID validation policy before a grounded renderer can cite chunks."""

from llm_grounding.domain.answers import (
    Assessment,
    DecisionAction,
    DecisionResult,
    ReasoningResult,
)
from llm_grounding.reasoning.basic_grounded import GroundedResponsePayload


class BasicGroundingPolicy:
    """Accept only schema-valid proposals whose nominated IDs passed deterministic validation."""

    async def decide(
        self, result: ReasoningResult, assessments: tuple[Assessment, ...]
    ) -> DecisionResult:
        """Reject unknown source references; honor explicit insufficient-evidence output."""
        payload = GroundedResponsePayload.model_validate(result.payload)
        reference = next(
            (item for item in assessments if item.dimension == "source_reference"), None
        )
        if reference is None or reference.label != "pass":
            return DecisionResult(
                DecisionAction.REJECT,
                result,
                {"reason": "unknown_or_unvalidated_source_references"},
            )
        if not payload.insufficient_evidence and not payload.claims:
            return DecisionResult(
                DecisionAction.REJECT,
                result,
                {"reason": "response_has_no_grounded_claims"},
            )
        action = (
            DecisionAction.ABSTAIN
            if payload.insufficient_evidence
            else DecisionAction.ACCEPT
        )
        return DecisionResult(
            action,
            result,
            {
                "reason": "insufficient_evidence"
                if action == DecisionAction.ABSTAIN
                else "validated_source_references"
            },
        )
