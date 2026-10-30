"""Score-normalized weighted fusion of independent dense and sparse retrievers."""

import asyncio
from dataclasses import replace

from llm_grounding.domain.corpus import ChunkRetrievalResult, RetrievalUnit
from llm_grounding.retrieval.models import (
    RetrievalContext,
    RetrievalRequest,
    RetrievalResponse,
)
from llm_grounding.retrieval.protocol import Retriever


class WeightedHybridRetriever:
    """Fuse dense and sparse scores after per-list min-max normalization."""

    def __init__(
        self,
        dense: Retriever,
        sparse: Retriever,
        *,
        dense_top_k: int = 20,
        sparse_top_k: int = 20,
        top_k: int = 10,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        score_normalization: str = "min_max",
    ) -> None:
        """Configure candidate limits and positive normalized-score weights."""
        if min(dense_top_k, sparse_top_k, top_k) < 1:
            raise ValueError("Hybrid retrieval limits must be positive")
        if dense_weight <= 0 or sparse_weight <= 0:
            raise ValueError("Hybrid retrieval weights must be positive")
        if score_normalization not in {"min_max", "none"}:
            raise ValueError("score_normalization must be min_max or none")
        self._dense = dense
        self._sparse = sparse
        self._dense_top_k = dense_top_k
        self._sparse_top_k = sparse_top_k
        self._top_k = top_k
        self._dense_weight = dense_weight
        self._sparse_weight = sparse_weight
        self._score_normalization = score_normalization

    async def retrieve(
        self,
        request: RetrievalRequest,
        context: RetrievalContext,
    ) -> RetrievalResponse:
        """Run both children concurrently and return deterministic weighted scores."""
        dense_response, sparse_response = await asyncio.gather(
            self._dense.retrieve(replace(request, top_k=self._dense_top_k), context),
            self._sparse.retrieve(replace(request, top_k=self._sparse_top_k), context),
        )
        normalize = self._score_normalization == "min_max"
        dense_scores = _normalized_scores(dense_response.results, normalize=normalize)
        sparse_scores = _normalized_scores(sparse_response.results, normalize=normalize)
        candidates: dict[str, tuple[RetrievalUnit, float, float]] = {}
        for result in dense_response.results:
            candidates[result.chunk.chunk_id] = (
                result.chunk,
                dense_scores[result.chunk.chunk_id],
                0.0,
            )
        for result in sparse_response.results:
            previous = candidates.get(result.chunk.chunk_id)
            candidates[result.chunk.chunk_id] = (
                result.chunk,
                previous[1] if previous else 0.0,
                sparse_scores[result.chunk.chunk_id],
            )
        fused = [
            (
                self._dense_weight * dense_score + self._sparse_weight * sparse_score,
                chunk,
                dense_score,
                sparse_score,
            )
            for chunk, dense_score, sparse_score in candidates.values()
        ]
        fused.sort(
            key=lambda item: (
                -item[0],
                item[1].document_id,
                item[1].order,
                item[1].chunk_id,
            )
        )
        results = tuple(
            ChunkRetrievalResult(chunk, rank, "hybrid_weighted", score)
            for rank, (score, chunk, _dense_score, _sparse_score) in enumerate(
                fused[: request.top_k or self._top_k], 1
            )
        )
        return RetrievalResponse(
            request.query,
            results,
            dense_response.query_embedding,
            metadata={
                "retriever": "hybrid_weighted",
                "score_type": "weighted_normalized_score",
                "normalization": self._score_normalization,
                "dense_weight": self._dense_weight,
                "sparse_weight": self._sparse_weight,
                "dense_candidate_count": len(dense_response.results),
                "sparse_candidate_count": len(sparse_response.results),
            },
        )


def _normalized_scores(
    results: tuple[ChunkRetrievalResult, ...], *, normalize: bool
) -> dict[str, float]:
    """Normalize scores to [0, 1], treating missing scores as zero."""
    values = {result.chunk.chunk_id: float(result.score or 0.0) for result in results}
    if not values:
        return {}
    if not normalize:
        return values
    minimum = min(values.values())
    maximum = max(values.values())
    if maximum == minimum:
        return {chunk_id: 1.0 for chunk_id in values}
    return {
        chunk_id: (score - minimum) / (maximum - minimum)
        for chunk_id, score in values.items()
    }
