"""PostgreSQL full-text retrieval over canonical persisted chunks."""

from llm_grounding.infrastructure.database.repositories import PostgresRepository
from llm_grounding.retrieval.models import (
    RetrievalContext,
    RetrievalRequest,
    RetrievalResponse,
)


class SparseRetriever:
    """Retrieve chunks with PostgreSQL ``websearch_to_tsquery`` and ``ts_rank_cd``."""

    def __init__(self, repository: PostgresRepository, top_k: int = 20) -> None:
        """Configure the canonical chunk repository and candidate limit."""
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self._repository = repository
        self._top_k = top_k

    async def retrieve(
        self,
        request: RetrievalRequest,
        context: RetrievalContext,
    ) -> RetrievalResponse:
        """Return project/document-filtered chunks ranked by native PostgreSQL FTS."""
        repository = context.repository or self._repository
        top_k = request.top_k or self._top_k
        results = await repository.search_chunks_fts(
            request.project_id,
            request.query,
            top_k,
            request.document_ids,
        )
        return RetrievalResponse(
            request.query,
            results,
            metadata={
                "retriever": "sparse",
                "score_type": "postgres_ts_rank_cd",
                "text_search_configuration": "simple",
            },
        )
