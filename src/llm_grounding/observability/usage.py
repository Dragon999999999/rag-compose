"""Aggregation of provider-reported and locally measured pipeline usage."""

from llm_grounding.observability.models import (
    LLMUsage,
    RagUsage,
    ResourceUsage,
    UsageOrigin,
)


class UsageAccumulator:
    """Aggregate target usage while retaining whether values are measured."""

    def __init__(self) -> None:
        """Start an empty accumulator."""
        self._llm: list[LLMUsage] = []
        self._embedding_calls = 0
        self._retrieval_calls = 0
        self._reranker_calls = 0
        self._context_tokens: int | None = None
        self._resources: list[ResourceUsage] = []

    def add_llm(self, usage: LLMUsage) -> None:
        """Add one model invocation, including evaluator roles for diagnostics."""
        self._llm.append(usage)

    def add_embedding(self, count: int = 1) -> None:
        """Increment embedding invocation count."""
        self._embedding_calls += count

    def add_retrieval(self, count: int = 1) -> None:
        """Increment retrieval invocation count."""
        self._retrieval_calls += count

    def add_reranker(self, count: int = 1) -> None:
        """Increment reranker invocation count."""
        self._reranker_calls += count

    def set_context_tokens(self, count: int | None) -> None:
        """Record context token count when a tokenizer actually measured it."""
        self._context_tokens = count

    def add_resource_usage(self, usage: ResourceUsage) -> None:
        """Retain an explicitly scoped local or provider-reported measurement."""
        self._resources.append(usage)

    def build(self) -> RagUsage:
        """Return an aggregate without inventing unavailable token values."""
        target = [item for item in self._llm if item.role == "target"]
        input_values = [item.input_tokens for item in target]
        output_values = [item.output_tokens for item in target]
        total_values = [item.total_tokens for item in target]
        input_tokens = _sum_complete(input_values)
        output_tokens = _sum_complete(output_values)
        total_tokens = _sum_complete(total_values)
        origins = {item.usage_origin for item in target}
        origin = (
            UsageOrigin.PROVIDER_REPORTED
            if origins == {UsageOrigin.PROVIDER_REPORTED}
            else UsageOrigin.UNAVAILABLE
            if not origins or origins == {UsageOrigin.UNAVAILABLE}
            else UsageOrigin.ESTIMATED
        )
        return RagUsage(
            llm_calls=len(target),
            embedding_calls=self._embedding_calls,
            retrieval_calls=self._retrieval_calls,
            reranker_calls=self._reranker_calls,
            context_tokens=self._context_tokens,
            total_input_tokens=input_tokens,
            total_output_tokens=output_tokens,
            total_tokens=total_tokens,
            usage_origin=origin,
            resource_usage=list(self._resources),
            metadata={
                "llm_usage": [item.model_dump(mode="json") for item in self._llm],
                "evaluator_llm_calls": len(self._llm) - len(target),
            },
        )


def _sum_complete(values: list[int | None]) -> int | None:
    """Sum token values only when every target invocation reported one."""
    if not values or any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)
