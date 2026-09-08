"""Produce candidate structured output from a task and selected evidence.

Reasoners own their output schema, may call models, and must surface model or
validation failures. Output is a proposal, not an acceptance or verification.
Claims may be absent. Implementations must not mutate source evidence or assume
a single invocation per run. See docs/architecture.md.
"""

from typing import Protocol

from llm_grounding.domain.answers import ReasoningResult
from llm_grounding.reasoning.models import ReasoningRequest


class ReasoningEngine(Protocol):
    """Define the interchangeable reasoning component contract."""

    async def reason(self, request: ReasoningRequest) -> ReasoningResult:
        """Propose schema-valid output and propagate generation failures."""
        ...
