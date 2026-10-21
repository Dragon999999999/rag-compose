"""Access-controlled, bounded raw-embedding diagnostics for evaluation."""

import os
from typing import Any, Protocol

from fastapi import APIRouter, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from llm_grounding.observability.embedding import EmbeddingRecord


class EmbeddingFetcher(Protocol):
    """Provide sampled vectors from a derived evaluation index."""

    async def fetch_vectors(
        self, chunk_ids: tuple[str, ...], *, limit: int = 128
    ) -> tuple[EmbeddingRecord, ...]:
        """Fetch bounded raw vectors without changing retrieval state."""
        ...


class EvalEmbeddingRequest(BaseModel):
    """Request a bounded set of evaluation-only vectors."""

    chunk_ids: list[str] = Field(min_length=1, max_length=128)
    include_vectors: bool = False


def create_eval_embeddings_router(
    fetcher: EmbeddingFetcher,
    *,
    access_token: str | None = None,
) -> APIRouter:
    """Create the optional embedding endpoint with explicit access control."""
    router = APIRouter(prefix="/eval/v1", tags=["evaluation-embeddings"])
    configured_token = access_token or os.getenv("LLM_GROUNDING_EVAL_EMBEDDING_TOKEN")

    @router.post("/embeddings", response_model=None)
    async def embeddings(
        body: EvalEmbeddingRequest,
        x_eval_embedding_token: str | None = Header(default=None),
    ) -> dict[str, Any] | JSONResponse:
        """Return sampled records, omitting vectors unless explicitly requested."""
        if not configured_token or x_eval_embedding_token != configured_token:
            return JSONResponse({"detail": "Embedding access is disabled."}, 403)
        records = await fetcher.fetch_vectors(tuple(body.chunk_ids), limit=128)
        if not body.include_vectors:
            records = tuple(
                record.model_copy(update={"vector": None}) for record in records
            )
        return {"records": [record.model_dump(mode="json") for record in records]}

    return router
