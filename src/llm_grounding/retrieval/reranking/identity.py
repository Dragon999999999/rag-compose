"""Preserve whole-document candidate selection unchanged."""

from llm_grounding.retrieval.models import RerankResponse


class IdentityReranker:
    """Perform no scoring, filtering, or reordering."""

    async def rerank(
        self, response: RerankResponse, *, top_k: int | None = None
    ) -> RerankResponse:
        """Return the original response including its ordering and query metadata."""
        if top_k is not None:
            if top_k < 1:
                raise ValueError("top_k must be positive")
            return RerankResponse(response.query, response.results[:top_k])
        return response
