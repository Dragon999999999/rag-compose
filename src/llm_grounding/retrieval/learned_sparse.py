"""Learned sparse retrieval backed by a lazily materialized Qdrant index."""

import asyncio
from typing import Protocol

from llm_grounding.domain.corpus import ChunkRetrievalResult, RetrievalUnit
from llm_grounding.embedding.models import SparseEmbedding
from llm_grounding.embedding.protocol import SparseEmbedder
from llm_grounding.infrastructure.database.repositories import PostgresRepository
from llm_grounding.retrieval.models import (
    RetrievalContext,
    RetrievalRequest,
    RetrievalResponse,
)


class SparseIndex(Protocol):
    """Minimal learned sparse index contract used by the retriever."""

    async def has_project_model(self, project_id: str) -> bool:
        """Return whether the project already has this model indexed."""
        ...

    async def upsert(
        self,
        project_id: str,
        chunks: tuple[RetrievalUnit, ...],
        vectors: tuple[SparseEmbedding, ...],
    ) -> None:
        """Persist one project-scoped sparse batch."""
        ...

    async def search(
        self,
        project_id: str,
        query: SparseEmbedding,
        top_k: int,
        document_ids: tuple[str, ...] = (),
    ) -> tuple[ChunkRetrievalResult, ...]:
        """Search a project-scoped sparse index."""
        ...


class LearnedSparseRetriever:
    """Index canonical chunks lazily and retrieve them with learned sparse weights."""

    def __init__(
        self,
        repository: PostgresRepository,
        embedder: SparseEmbedder,
        index: SparseIndex,
        *,
        retriever_id: str,
        top_k: int = 20,
    ) -> None:
        """Configure a model-specific sparse embedder and persistent index."""
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self._repository = repository
        self._embedder = embedder
        self._index = index
        self._retriever_id = retriever_id
        self._top_k = top_k
        self._project_locks: dict[str, asyncio.Lock] = {}

    async def retrieve(
        self,
        request: RetrievalRequest,
        context: RetrievalContext,
    ) -> RetrievalResponse:
        """Ensure the project index exists, encode the query, and search it."""
        repository = context.repository or self._repository
        await self._ensure_index(request.project_id, repository)
        query = await self._embedder.embed_sparse_query(request.query)
        results = await self._index.search(
            request.project_id,
            query,
            request.top_k or self._top_k,
            request.document_ids,
        )
        return RetrievalResponse(
            request.query,
            tuple(
                ChunkRetrievalResult(
                    result.chunk, rank, self._retriever_id, result.score
                )
                for rank, result in enumerate(results, 1)
            ),
            metadata={
                "retriever": self._retriever_id,
                "score_type": "sparse_dot_product",
                "model_identity": self._embedder.model_identity,
            },
        )

    async def _ensure_index(
        self, project_id: str, repository: PostgresRepository
    ) -> None:
        """Build one project index once, sharing concurrent first-use requests."""
        lock = self._project_locks.setdefault(project_id, asyncio.Lock())
        async with lock:
            if await self._index.has_project_model(project_id):
                return
            chunks = await repository.chunks_for_project(project_id)
            if not chunks:
                return
            vectors = await self._embedder.embed_sparse_documents(
                [chunk.text for chunk in chunks]
            )
            await self._index.upsert(project_id, chunks, vectors)


class BgeM3SparseRetriever(LearnedSparseRetriever):
    """Learned sparse retrieval using the existing BGE-M3 model identity."""

    def __init__(self, *args: object, top_k: int = 20, **kwargs: object) -> None:
        """Set the stable BGE-M3 retriever identifier."""
        super().__init__(*args, retriever_id="bge_m3_sparse", top_k=top_k, **kwargs)  # type: ignore[arg-type]
