"""Lazy local Sentence Transformers cross-encoder reranking."""

import math
import os
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from llm_grounding.retrieval.models import RetrievalResponse

DEFAULT_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
DEFAULT_CACHE_DIR = Path("~/.cache/rag-eval/models")


class CrossEncoderReranker:
    """Score query/candidate pairs with a lazily loaded local cross-encoder.

    The optional ``sentence-transformers`` dependency is imported only on first
    inference. Model files are cached below ``RAG_EVAL_MODEL_CACHE_DIR`` when set,
    otherwise below ``~/.cache/rag-eval/models``. The loaded model is reused for
    the lifetime of this reranker instance.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "cpu",
        batch_size: int = 16,
        top_k: int | None = None,
        *,
        cache_dir: Path | None = None,
        model_factory: Callable[..., object] | None = None,
    ) -> None:
        """Validate local inference settings without importing or loading a model."""
        if not model_name.strip():
            raise ValueError("model_name must be non-empty")
        if not device.strip():
            raise ValueError("device must be non-empty")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if top_k is not None and top_k < 1:
            raise ValueError("rerank_top_k must be positive")
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._top_k = top_k
        configured_cache = os.environ.get("RAG_EVAL_MODEL_CACHE_DIR", "").strip()
        self._cache_dir = (
            cache_dir or Path(configured_cache or DEFAULT_CACHE_DIR).expanduser()
        )
        self._model_factory = model_factory
        self._model: object | None = None

    @property
    def model(self) -> object | None:
        """Expose the loaded model for diagnostics without forcing lazy loading."""
        return self._model

    async def rerank(
        self, response: RetrievalResponse, *, top_k: int | None = None
    ) -> RetrievalResponse:
        """Score candidates in descending order while preserving metadata."""
        if not response.results:
            return response
        limit = top_k if top_k is not None else self._top_k
        if limit is not None and limit < 1:
            raise ValueError("rerank_top_k must be positive")
        # CrossEncoder.predict is synchronous; keeping the call inside the async
        # boundary matches the existing provider contract and avoids creating a
        # second executor lifecycle for each small reranking request.
        scored = self._score(response.query, response.results)
        ordered = sorted(
            zip(response.results, scored, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )
        if limit is not None:
            ordered = ordered[:limit]
        return RetrievalResponse(
            response.query,
            tuple(
                replace(result, rank=rank, score=score)
                for rank, (result, score) in enumerate(ordered, 1)
            ),
        )

    def _score(
        self,
        query: str,
        results: Sequence[Any],
    ) -> tuple[float, ...]:
        """Load the model on demand and score all pairs in one configured batch."""
        model = self._load_model()
        pairs = [[query, result.evidence.text or ""] for result in results]
        scores = model.predict(  # type: ignore[attr-defined]
            pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
        )
        values = tuple(self._score_value(score) for score in scores)
        if len(values) != len(results):
            raise RuntimeError("Cross-encoder returned an unexpected score count")
        return values

    def _load_model(self) -> object:
        """Construct the optional model once and create its persistent cache path."""
        if self._model is not None:
            return self._model
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        if self._model_factory is not None:
            self._model = self._model_factory(
                self._model_name,
                device=self._device,
                cache_folder=str(self._cache_dir),
            )
            return self._model
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "Install llm-grounding[reranking] to use cross_encoder reranking"
            ) from exc
        self._model = CrossEncoder(
            self._model_name,
            device=self._device,
            cache_folder=str(self._cache_dir),
        )
        return self._model

    @staticmethod
    def _score_value(value: object) -> float:
        """Convert provider scalar output into a finite Python float."""
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise RuntimeError("Cross-encoder returned a non-scalar score")
            value = value[0]
        item = getattr(value, "item", None)
        if callable(item):
            value = item()
        try:
            score = float(cast(Any, value))
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Cross-encoder returned a non-numeric score") from exc
        if not math.isfinite(score):
            raise RuntimeError("Cross-encoder returned a non-finite score")
        return score
