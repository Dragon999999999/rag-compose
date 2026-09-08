"""Baseline assessment that makes no grounding or correctness judgments."""

from llm_grounding.domain.answers import Assessment, ReasoningResult
from llm_grounding.domain.evidence import RetrievalResult


class NoOpAssessor:
    """Return no assessments; this is not evidence of verification success."""

    async def assess(
        self, result: ReasoningResult, evidence: tuple[RetrievalResult, ...]
    ) -> tuple[Assessment, ...]:
        """Leave the candidate unassessed, regardless of selected evidence."""
        return ()
