"""PostgreSQL-backed retrieval that does not require an embedding index."""

from math import ceil

from llm_grounding.domain.corpus import ChunkRetrievalResult
from llm_grounding.infrastructure.database.repositories import PostgresRepository
from llm_grounding.retrieval.models import (
    RetrievalContext,
    RetrievalRequest,
    RetrievalResponse,
)


def _estimated_tokens(text: str) -> int:
    """Estimate tokens conservatively without loading a tokenizer dependency."""
    return max(1, ceil(len(text) / 4))


class FullContextRetriever:
    """Return project chunks in persisted order within an explicit token budget.

    The repository remains the source of chunk text and provenance. When the
    project is larger than the configured budget, chunks are selected from the
    beginning of the stable project/document order and all omitted identifiers
    are reported in the response metadata. No embedding provider or vector
    backend is constructed or consulted.
    """

    def __init__(
        self,
        repository: PostgresRepository,
        context_budget_tokens: int = 16_000,
    ) -> None:
        """Configure the canonical chunk repository and positive budget."""
        if context_budget_tokens < 1:
            raise ValueError("context_budget_tokens must be positive")
        self._repository = repository
        self._context_budget_tokens = context_budget_tokens

    async def retrieve(
        self,
        request: RetrievalRequest,
        context: RetrievalContext,
    ) -> RetrievalResponse:
        """Load project chunks and report any budget-driven omissions explicitly."""
        repository = context.repository or self._repository
        chunks = await repository.chunks_for_project(
            request.project_id, request.document_ids
        )
        budget = self._context_budget_tokens
        selected: list[ChunkRetrievalResult] = []
        omitted_ids: list[str] = []
        used_tokens = 0
        for chunk in chunks:
            if request.top_k is not None and len(selected) >= request.top_k:
                omitted_ids.append(chunk.chunk_id)
                continue
            chunk_tokens = _estimated_tokens(chunk.text)
            if used_tokens + chunk_tokens > budget:
                omitted_ids.append(chunk.chunk_id)
                continue
            selected.append(
                ChunkRetrievalResult(
                    chunk,
                    len(selected) + 1,
                    "full_context",
                    None,
                )
            )
            used_tokens += chunk_tokens
        return RetrievalResponse(
            request.query,
            tuple(selected),
            metadata={
                "retriever": "full_context",
                "score_type": "none",
                "context_budget_tokens": budget,
                "estimated_tokens": used_tokens,
                "total_chunks": len(chunks),
                "included_chunks": len(selected),
                "omitted_chunks": len(omitted_ids),
                "omitted_chunk_ids": omitted_ids,
                "budget_exceeded": bool(omitted_ids),
            },
        )
