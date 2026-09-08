"""Measure terminal runs without participating in acceptance decisions.

Evaluators consume completed or failed RagRun records and return named metrics.
They must not mutate run artifacts or imply scientific validity from structural
success. Implementations may perform external I/O and should surface evaluation
failures. This contract is separate from in-run Assessor measurements; see
docs/architecture.md for the experiment lifecycle.
"""

from typing import Protocol

from llm_grounding.domain.runs import RagRun
from llm_grounding.evaluation.models import MetricResult


class Evaluator(Protocol):
    """Measure completed architecture invocations for experiments."""

    async def evaluate(self, run: RagRun) -> tuple[MetricResult, ...]:
        """Return metrics for a terminal run; raise ValueError for an active run."""
        ...
