"""Select evidence candidates for a query without producing an answer.

Implementations return ordered candidates with query-specific ranks and optional
scores. They may perform I/O and must surface retrieval failures. Empty results
are permitted. Canonical evidence must remain unchanged. Calls are independent;
protocols do not limit the number of retrievals in a run. See docs/architecture.md.
"""

from typing import Protocol

from llm_grounding.retrieval.models import (
    RetrievalContext,
    RetrievalRequest,
    RetrievalResponse,
)


class Retriever(Protocol):
    """Select project evidence from resolved runtime resources.

    Implementations must return the shared chunk/provenance result shape and may
    perform I/O through the supplied context. They must not construct providers
    or silently substitute unavailable resources.
    """

    async def retrieve(
        self, request: RetrievalRequest, context: RetrievalContext
    ) -> RetrievalResponse:
        """Return ranked candidates for the project-scoped request."""
        ...
