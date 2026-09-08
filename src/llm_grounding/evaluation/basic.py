"""Structural baseline evaluation, without scientific grounding quality metrics."""

from pydantic import ValidationError

from llm_grounding.domain.runs import RagRun
from llm_grounding.evaluation.models import MetricResult
from llm_grounding.reasoning.basic_summary import BasicSummaryPayload


class BasicPipelineEvaluator:
    """Check successful completion and baseline schema validity only."""

    async def evaluate(self, run: RagRun) -> tuple[MetricResult, ...]:
        """Measure a terminal run; failure yields a failed structural metric."""
        if run.status not in {"completed", "failed"}:
            raise ValueError("Evaluation requires a completed or failed run.")
        succeeded = run.status == "completed" and run.answer is not None
        valid = False
        if run.answer is not None:
            try:
                payload = BasicSummaryPayload.model_validate(run.answer.payload)
                valid = bool(payload.summary.strip())
            except ValidationError:
                valid = False
        return (
            MetricResult(
                "pipeline_succeeded",
                succeeded,
                run.run_id,
                "Final answer exists after completion.",
            ),
            MetricResult(
                "baseline_schema_valid",
                valid,
                run.run_id,
                "5–8 keywords and a 20–500 character summary; no quality judgment.",
            ),
        )
