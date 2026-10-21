"""Preserve whole-document candidate selection unchanged."""

from llm_grounding.retrieval.models import RetrievalResponse


class IdentityReranker:
    """Perform no scoring, filtering, or reordering."""

    async def rerank(
        self, response: RetrievalResponse, *, top_k: int | None = None
    ) -> RetrievalResponse:
        """Return the original response including its ordering and query metadata."""
        if top_k is not None:
            if top_k < 1:
                raise ValueError("top_k must be positive")
            return RetrievalResponse(response.query, response.results[:top_k])
        return response
