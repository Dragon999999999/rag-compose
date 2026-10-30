"""Refine existing retrieval candidates without retrieving new evidence.

Implementations may reorder or filter candidates, maintaining query identity and
consistent one-based ranks. Canonical evidence remains unchanged. Model-based
implementations may perform I/O and must propagate failures. Empty input is valid.
"""

from typing import Protocol

from llm_grounding.retrieval.models import RerankResponse


class Reranker(Protocol):
    """Define the interchangeable reranking component contract."""

    async def rerank(
        self, response: RerankResponse, *, top_k: int | None = None
    ) -> RerankResponse:
        """Return refined candidates drawn only from the supplied response.

        Args:
            response: Query and candidates produced by the retrieval stage.
            top_k: Optional request-local output limit overriding provider config.
        """
        ...
