"""Preserve whole-document candidate selection unchanged."""

from llm_grounding.retrieval.models import RetrievalResponse


class IdentityReranker:
    """Perform no scoring, filtering, or reordering."""

    async def rerank(self, response: RetrievalResponse) -> RetrievalResponse:
        """Return the original response including its ordering and query metadata."""
        return response
