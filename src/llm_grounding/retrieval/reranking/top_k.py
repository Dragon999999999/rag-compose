"""Lightweight candidate truncation without a relevance model."""

from dataclasses import replace

from llm_grounding.retrieval.models import RetrievalResponse


class TopKReranker:
    """Keep the existing retrieval order and return at most a configured limit."""

    def __init__(self, top_k: int | None = None) -> None:
        """Configure an optional final candidate limit."""
        if top_k is not None and top_k < 1:
            raise ValueError("rerank_top_k must be positive")
        self._top_k = top_k

    async def rerank(
        self, response: RetrievalResponse, *, top_k: int | None = None
    ) -> RetrievalResponse:
        """Truncate candidates while preserving identifiers, metadata, and order."""
        limit = top_k if top_k is not None else self._top_k
        if limit is not None and limit < 1:
            raise ValueError("rerank_top_k must be positive")
        selected = response.results if limit is None else response.results[:limit]
        results = tuple(
            replace(result, rank=rank) for rank, result in enumerate(selected, 1)
        )
        if results == response.results:
            return response
        return RetrievalResponse(response.query, results)
