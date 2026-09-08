"""Unconditional acceptance for the unverified extraction baseline."""

from llm_grounding.domain.answers import (
    Assessment,
    DecisionAction,
    DecisionResult,
    ReasoningResult,
)


class AcceptAllDecisionPolicy:
    """Authorize every candidate unchanged, without asserting its correctness."""

    async def decide(
        self, result: ReasoningResult, assessments: tuple[Assessment, ...]
    ) -> DecisionResult:
        """Accept candidate output even when no substantive assessments exist."""
        return DecisionResult(DecisionAction.ACCEPT, result)
