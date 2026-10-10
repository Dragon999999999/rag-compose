"""Technology boundary for schema-constrained model generation.

Callers supply a prompt and Pydantic output schema; implementations return a
validated instance of that schema. Adapters may perform network I/O, must bound
timeouts/retries, and must propagate failures without exposing credentials or
provider response bodies. Model configuration is constructor-injected. This
contract does not couple reasoners to PydanticAI or a vendor client lifecycle.
"""

from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel

from llm_grounding.observability.models import LLMUsage


@dataclass(frozen=True)
class GenerationResult[OutputT: BaseModel]:
    """Validated model output plus provider usage and model identity."""

    output: "OutputT"
    usage: LLMUsage | None = None
    model_id: str | None = None

    def __eq__(self, other: object) -> bool:
        """Remain value-compatible with legacy callers comparing raw output."""
        if isinstance(other, GenerationResult):
            return (
                self.output == other.output
                and self.usage == other.usage
                and self.model_id == other.model_id
            )
        return self.output == other


class StructuredModelClient(Protocol):
    """Generate typed structured output for independently composed reasoners."""

    async def generate[OutputT: BaseModel](
        self,
        prompt: str,
        output_type: type[OutputT],
        *,
        model_name: str | None = None,
    ) -> GenerationResult[OutputT]:
        """Return validated output and usage, or raise without partial JSON."""
        ...
