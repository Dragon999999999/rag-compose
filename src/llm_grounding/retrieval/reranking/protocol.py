"""Refine existing retrieval candidates without retrieving new evidence.

Implementations may reorder or filter candidates, maintaining query identity and
consistent one-based ranks. Canonical evidence remains unchanged. Model-based
implementations may perform I/O and must propagate failures. Empty input is valid.
"""

from typing import Protocol

from llm_grounding.retrieval.models import RetrievalResponse


class Reranker(Protocol):
    """Define the interchangeable reranking component contract."""

    async def rerank(self, response: RetrievalResponse) -> RetrievalResponse:
        """Return refined candidates drawn only from the supplied response."""
        ...
