"""Select evidence candidates for a query without producing an answer.

Implementations return ordered candidates with query-specific ranks and optional
scores. They may perform I/O and must surface retrieval failures. Empty results
are permitted. Canonical evidence must remain unchanged. Calls are independent;
protocols do not limit the number of retrievals in a run. See docs/architecture.md.
"""

from typing import Protocol

from llm_grounding.retrieval.models import RetrievalRequest, RetrievalResponse


class Retriever(Protocol):
    """Define the interchangeable retrieval component contract."""

    async def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        """Return candidates using the documented selection and ordering rules."""
        ...
