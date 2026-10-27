"""Local-model reranking through the shared model-management lifecycle."""

from dataclasses import replace

from llm_grounding.infrastructure.model_management.manager import LocalModelManager
from llm_grounding.retrieval.models import RerankResponse


class LocalModelReranker:
    """Rerank candidates with a registered model leased on demand.

    The manager owns downloading, runtime startup, reuse, and idle shutdown.
    This adapter only translates canonical evidence candidates into the local
    ``/rerank`` contract and translates returned scores back into the canonical
    response envelope.
    """

    def __init__(
        self,
        model_id: str,
        manager: LocalModelManager,
        *,
        top_k: int | None = None,
    ) -> None:
        """Configure a model reference without acquiring or loading it."""
        if not model_id.strip():
            raise ValueError("model_id must be non-empty")
        if top_k is not None and top_k < 1:
            raise ValueError("rerank_top_k must be positive")
        self._model_id = model_id
        self._manager = manager
        self._top_k = top_k

    @property
    def model_id(self) -> str:
        """Return the registered model ID used for inference."""
        return self._model_id

    async def rerank(
        self, response: RerankResponse, *, top_k: int | None = None
    ) -> RerankResponse:
        """Score candidates and preserve all evidence and retrieval metadata."""
        if not response.results:
            return response
        limit = top_k if top_k is not None else self._top_k
        if limit is not None and limit < 1:
            raise ValueError("rerank_top_k must be positive")
        documents = [result.evidence.text or "" for result in response.results]
        async with self._manager.acquire(self._model_id) as model:
            scores = await model.rerank(response.query, documents)
        if len(scores) != len(response.results):
            raise RuntimeError("Local reranker returned an unexpected score count")
        indexed_scores = {index: score for index, score in scores}
        if set(indexed_scores) != set(range(len(response.results))):
            raise RuntimeError("Local reranker returned invalid candidate indices")
        ordered = sorted(
            enumerate(response.results),
            key=lambda item: (-indexed_scores[item[0]], item[1].rank, item[0]),
        )
        if limit is not None:
            ordered = ordered[:limit]
        return RerankResponse(
            response.query,
            tuple(
                replace(
                    result,
                    rank=rank,
                    score=indexed_scores[index],
                )
                for rank, (index, result) in enumerate(ordered, 1)
            ),
        )
