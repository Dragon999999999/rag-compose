"""Explicit experimental execution; normal user requests never run evaluators."""

from llm_grounding.evaluation.models import CaseEvaluation, MetricResult, TestSuite
from llm_grounding.evaluation.protocol import Evaluator
from llm_grounding.orchestration.runner import ArchitectureRunner, RunFailedError


class EvaluationRunner:
    """Execute a suite and evaluate each terminal run, retaining failed cases."""

    def __init__(
        self, runner: ArchitectureRunner, evaluators: tuple[Evaluator, ...]
    ) -> None:
        """Inject the architecture execution strategy and experimental measurements."""
        self._runner = runner
        self._evaluators = evaluators

    async def run(self, suite: TestSuite) -> tuple[CaseEvaluation, ...]:
        """Execute cases sequentially and propagate evaluator failures."""
        evaluations: list[CaseEvaluation] = []
        for case in suite.cases:
            try:
                run = await self._runner.run(
                    case.request, {"test_suite": suite.name, "test_case": case.case_id}
                )
            except RunFailedError as exc:
                run = exc.run
            metrics: list[MetricResult] = []
            for evaluator in self._evaluators:
                metrics.extend(await evaluator.evaluate(run))
            evaluations.append(CaseEvaluation(case.case_id, run, tuple(metrics)))
        return tuple(evaluations)
