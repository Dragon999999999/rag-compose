"""ColBERT-style late-interaction retrieval with MaxSim scoring."""

import asyncio
from typing import Protocol

from llm_grounding.domain.corpus import ChunkRetrievalResult, RetrievalUnit
from llm_grounding.embedding.models import TokenEmbedding
from llm_grounding.embedding.protocol import TokenEmbedder
from llm_grounding.infrastructure.database.repositories import PostgresRepository
from llm_grounding.retrieval.models import (
    RetrievalContext,
    RetrievalRequest,
    RetrievalResponse,
)


class MultiVectorIndex(Protocol):
    """Minimal token-matrix index contract used by ColBERT retrieval."""

    async def has_project_model(self, project_id: str) -> bool:
        """Return whether the project already has this model indexed."""
        ...

    async def upsert(
        self,
        project_id: str,
        chunks: tuple[RetrievalUnit, ...],
        vectors: tuple[TokenEmbedding, ...],
    ) -> None:
        """Persist one project-scoped token batch."""
        ...

    async def search(
        self,
        project_id: str,
        query: TokenEmbedding,
        top_k: int,
        document_ids: tuple[str, ...] = (),
    ) -> tuple[ChunkRetrievalResult, ...]:
        """Search a project-scoped MaxSim index."""
        ...


def maxsim_score(query: TokenEmbedding, document: TokenEmbedding) -> float:
    """Compute ColBERT MaxSim: sum over query tokens of best document similarity."""
    if not query.values or not document.values:
        return 0.0
    if query.dimension != document.dimension:
        raise ValueError("ColBERT query and document dimensions must match")
    return sum(
        max(
            sum(query_token[i] * document_token[i] for i in range(query.dimension))
            for document_token in document.values
        )
        for query_token in query.values
    )


class ColBERTRetriever:
    """Use BM25 candidates and a persistent Qdrant MaxSim index."""

    def __init__(
        self,
        repository: PostgresRepository,
        embedder: TokenEmbedder,
        index: MultiVectorIndex,
        *,
        candidate_top_k: int = 50,
        top_k: int = 10,
    ) -> None:
        """Configure bounded candidate retrieval and token-level reranking."""
        if candidate_top_k < 1 or top_k < 1:
            raise ValueError("ColBERT limits must be positive")
        self._repository = repository
        self._embedder = embedder
        self._index = index
        self._candidate_top_k = candidate_top_k
        self._top_k = top_k
        self._locks: dict[str, asyncio.Lock] = {}

    async def retrieve(
        self,
        request: RetrievalRequest,
        context: RetrievalContext,
    ) -> RetrievalResponse:
        """Materialize token vectors on first use and run MaxSim search."""
        repository = context.repository or self._repository
        await self._ensure_index(request.project_id, repository, request.document_ids)
        query = await self._embedder.embed_token_query(request.query)
        results = await self._index.search(
            request.project_id,
            query,
            request.top_k or self._top_k,
            request.document_ids,
        )
        return RetrievalResponse(
            request.query,
            tuple(
                ChunkRetrievalResult(result.chunk, rank, "colbert_v2", result.score)
                for rank, result in enumerate(results, 1)
            ),
            metadata={
                "retriever": "colbert_v2",
                "score_type": "maxsim",
                "candidate_top_k": self._candidate_top_k,
                "model_identity": self._embedder.model_identity,
            },
        )

    async def _ensure_index(
        self,
        project_id: str,
        repository: PostgresRepository,
        document_ids: tuple[str, ...],
    ) -> None:
        """Index only project chunks selected by the candidate stage."""
        lock = self._locks.setdefault(project_id, asyncio.Lock())
        async with lock:
            if await self._index.has_project_model(project_id):
                return
            chunks = await repository.chunks_for_project(project_id, document_ids)
            if not chunks:
                return
            vectors = await self._embedder.embed_token_documents(
                [chunk.text for chunk in chunks]
            )
            await self._index.upsert(project_id, chunks, vectors)
