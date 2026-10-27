"""Retriever-level dense and PostgreSQL sparse Reciprocal Rank Fusion."""

from dataclasses import replace

from llm_grounding.domain.corpus import ChunkRetrievalResult, RetrievalUnit
from llm_grounding.retrieval.models import (
    RetrievalContext,
    RetrievalRequest,
    RetrievalResponse,
)
from llm_grounding.retrieval.protocol import Retriever


class HybridRetriever:
    """Fuse dense and sparse candidates without native vector hybrid search."""

    def __init__(
        self,
        dense: Retriever,
        sparse: Retriever,
        *,
        dense_top_k: int = 20,
        sparse_top_k: int = 20,
        top_k: int = 10,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ) -> None:
        """Configure child retrievers and validated weighted-RRF parameters."""
        if min(dense_top_k, sparse_top_k, top_k, rrf_k) < 1:
            raise ValueError("Hybrid retrieval limits and rrf_k must be positive")
        if dense_weight <= 0 or sparse_weight <= 0:
            raise ValueError("Hybrid retrieval weights must be positive")
        self._dense = dense
        self._sparse = sparse
        self._dense_top_k = dense_top_k
        self._sparse_top_k = sparse_top_k
        self._top_k = top_k
        self._rrf_k = rrf_k
        self._dense_weight = dense_weight
        self._sparse_weight = sparse_weight

    async def retrieve(
        self,
        request: RetrievalRequest,
        context: RetrievalContext,
    ) -> RetrievalResponse:
        """Run both strategies and return stable-ID deduplicated RRF candidates."""
        dense_response, sparse_response = await _gather_retrievals(
            self._dense,
            self._sparse,
            replace(request, top_k=self._dense_top_k),
            replace(request, top_k=self._sparse_top_k),
            context,
        )
        candidates: dict[str, dict[str, object]] = {}
        for result in dense_response.results:
            candidates.setdefault(result.chunk.chunk_id, {"chunk": result.chunk})[
                "dense"
            ] = result
        for result in sparse_response.results:
            candidates.setdefault(result.chunk.chunk_id, {"chunk": result.chunk})[
                "sparse"
            ] = result

        fused: list[tuple[float, str, RetrievalUnit]] = []
        component_metadata: dict[str, dict[str, object]] = {}
        for chunk_id, candidate in candidates.items():
            dense_result = candidate.get("dense")
            sparse_result = candidate.get("sparse")
            dense_rank = getattr(dense_result, "rank", None)
            sparse_rank = getattr(sparse_result, "rank", None)
            fused_score = 0.0
            if isinstance(dense_rank, int):
                fused_score += self._dense_weight / (self._rrf_k + dense_rank)
            if isinstance(sparse_rank, int):
                fused_score += self._sparse_weight / (self._rrf_k + sparse_rank)
            chunk = candidate["chunk"]
            if not isinstance(chunk, RetrievalUnit):
                raise TypeError("Hybrid child retrievers returned an invalid chunk")
            component_metadata[chunk_id] = {
                "dense_score": getattr(dense_result, "score", None),
                "dense_rank": dense_rank,
                "sparse_score": getattr(sparse_result, "score", None),
                "sparse_rank": sparse_rank,
            }
            fused.append((fused_score, chunk_id, chunk))
        fused.sort(
            key=lambda item: (-item[0], item[2].document_id, item[2].order, item[1])
        )
        results = tuple(
            _hybrid_result(chunk, rank, score)
            for rank, (score, _chunk_id, chunk) in enumerate(
                fused[: request.top_k or self._top_k], 1
            )
        )
        return RetrievalResponse(
            request.query,
            results,
            dense_response.query_embedding,
            metadata={
                "retriever": "hybrid",
                "score_type": "rrf",
                "rrf_k": self._rrf_k,
                "dense_weight": self._dense_weight,
                "sparse_weight": self._sparse_weight,
                "dense_candidate_count": len(dense_response.results),
                "sparse_candidate_count": len(sparse_response.results),
                "component_scores": component_metadata,
            },
        )


async def _gather_retrievals(
    dense: Retriever,
    sparse: Retriever,
    dense_request: RetrievalRequest,
    sparse_request: RetrievalRequest,
    context: RetrievalContext,
) -> tuple[RetrievalResponse, RetrievalResponse]:
    """Run child searches concurrently while preserving their canonical envelopes."""
    import asyncio

    responses = await asyncio.gather(
        dense.retrieve(dense_request, context),
        sparse.retrieve(sparse_request, context),
    )
    return responses[0], responses[1]


def _hybrid_result(
    chunk: RetrievalUnit, rank: int, score: float
) -> ChunkRetrievalResult:
    """Build a canonical fused result."""
    return ChunkRetrievalResult(chunk, rank, "hybrid", score)
