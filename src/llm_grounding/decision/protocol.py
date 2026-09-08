"""Authorize, reject, or abstain on candidate output using assessments.

Policies act on judgments without inventing new measurements. Renderers consume
the explicit action and may use only authorized output. Implementations may
perform I/O where needed and must surface policy failures. Empty assessments are
valid input but confer no verification guarantee. See docs/architecture.md.
"""

from typing import Protocol

from llm_grounding.domain.answers import Assessment, DecisionResult, ReasoningResult


class DecisionPolicy(Protocol):
    """Define the interchangeable decision component contract."""

    async def decide(
        self, result: ReasoningResult, assessments: tuple[Assessment, ...]
    ) -> DecisionResult:
        """Apply policy and explicitly state whether the result may be rendered."""
        ...
