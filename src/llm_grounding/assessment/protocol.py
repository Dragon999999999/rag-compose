"""Measure candidate properties inside a run independently of acceptance policy.

Assessors receive candidate output and selected evidence and return measurements
with target IDs and dimensions. No measurements means no judgment was made, not
that verification passed. Implementations may perform I/O and must surface
failures. They must not change the candidate or apply acceptance thresholds.
"""

from typing import Protocol

from llm_grounding.domain.answers import Assessment, ReasoningResult
from llm_grounding.domain.evidence import RetrievalResult


class Assessor(Protocol):
    """Define the interchangeable assessment component contract."""

    async def assess(
        self, result: ReasoningResult, evidence: tuple[RetrievalResult, ...]
    ) -> tuple[Assessment, ...]:
        """Return measurements of the candidate; an empty tuple means no assessment."""
        ...
