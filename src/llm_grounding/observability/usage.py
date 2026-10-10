"""Aggregation of provider-reported and locally measured pipeline usage."""

from llm_grounding.observability.models import LLMUsage, RagUsage, UsageOrigin


class UsageAccumulator:
    """Aggregate target usage while retaining whether values are measured."""

    def __init__(self) -> None:
        """Start an empty accumulator."""
        self._llm: list[LLMUsage] = []
        self._embedding_calls = 0
        self._retrieval_calls = 0
        self._reranker_calls = 0
        self._context_tokens: int | None = None

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

    def build(self) -> RagUsage:
        """Return an aggregate without inventing unavailable token values."""
        target = [item for item in self._llm if item.role == "target"]
        input_values = [item.input_tokens for item in target]
        output_values = [item.output_tokens for item in target]
        total_values = [item.total_tokens for item in target]
        input_tokens = (
            sum(input_values)
            if all(value is not None for value in input_values) and input_values
            else None
        )
        output_tokens = (
            sum(output_values)
            if all(value is not None for value in output_values) and output_values
            else None
        )
        total_tokens = (
            sum(total_values)
            if all(value is not None for value in total_values) and total_values
            else None
        )
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
            metadata={
                "llm_usage": [item.model_dump(mode="json") for item in self._llm],
                "evaluator_llm_calls": len(self._llm) - len(target),
            },
        )
