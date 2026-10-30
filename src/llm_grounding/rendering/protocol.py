"""Present policy-authorized structured results as general answers.

Renderers must respect the decision action and never independently authorize
rejected content. They validate payload compatibility and surface mismatches.
Rendering need not use a model or perform I/O. Terminal presentation belongs to
the application.
"""

from typing import Protocol

from llm_grounding.domain.answers import Answer, DecisionResult


class Renderer(Protocol):
    """Define the interchangeable rendering component contract."""

    async def render(self, decision: DecisionResult) -> Answer:
        """Render authorized output or raise ValueError for unsupported decisions."""
        ...
