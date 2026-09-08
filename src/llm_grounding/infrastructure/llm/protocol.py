"""Technology boundary for schema-constrained model generation.

Callers supply a prompt and Pydantic output schema; implementations return a
validated instance of that schema. Adapters may perform network I/O, must bound
timeouts/retries, and must propagate failures without exposing credentials or
provider response bodies. Model configuration is constructor-injected. This
contract does not couple reasoners to PydanticAI or a vendor client lifecycle.
"""

from typing import Protocol

from pydantic import BaseModel


class StructuredModelClient(Protocol):
    """Generate typed structured output for independently composed reasoners."""

    async def generate[OutputT: BaseModel](
        self, prompt: str, output_type: type[OutputT]
    ) -> OutputT:
        """Return schema-valid output or raise an error; never return partial JSON."""
        ...
