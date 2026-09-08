"""Deterministic presentation of accepted basic summary payloads."""

from llm_grounding.domain.answers import Answer, DecisionAction, DecisionResult
from llm_grounding.reasoning.basic_summary import BasicSummaryPayload


class BasicSummaryRenderer:
    """Render keywords and summary without model calls or acceptance decisions."""

    async def render(self, decision: DecisionResult) -> Answer:
        """Validate an accepted payload and render it; reject incompatible actions."""
        if decision.action != DecisionAction.ACCEPT:
            raise ValueError("Basic summary renderer requires an accepted decision.")
        payload = BasicSummaryPayload.model_validate(decision.reasoning.payload)
        keywords = "\n".join(
            f"{index}. {keyword}"
            for index, keyword in enumerate(payload.keywords, start=1)
        )
        return Answer(
            text=f"Keywords:\n{keywords}\n\nSummary:\n{payload.summary}",
            payload=payload.model_dump(mode="json"),
            claims=decision.reasoning.claims,
            metadata={"verified": False},
        )
