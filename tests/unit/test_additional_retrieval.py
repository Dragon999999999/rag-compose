"""Verify the additional retrieval contracts without external services."""

from dataclasses import dataclass

import pytest

from llm_grounding.domain.corpus import ChunkRetrievalResult, RetrievalUnit
from llm_grounding.embedding.models import SparseEmbedding, TokenEmbedding
from llm_grounding.retrieval.bm25 import BM25Retriever
from llm_grounding.retrieval.colbert import maxsim_score
from llm_grounding.retrieval.hybrid_weighted import WeightedHybridRetriever
from llm_grounding.retrieval.learned_sparse import BgeM3SparseRetriever
from llm_grounding.retrieval.models import (
    RetrievalContext,
    RetrievalRequest,
    RetrievalResponse,
)


def _chunk(chunk_id: str, text: str, order: int) -> RetrievalUnit:
    """Build a minimal canonical chunk for retrieval unit tests."""
    return RetrievalUnit(
        chunk_id, "document", "ingestion", "paper.pdf", text, order, None, None, ()
    )


class _Repository:
    """In-memory repository subset used by retriever tests."""

    def __init__(self, chunks: tuple[RetrievalUnit, ...]) -> None:
        self.chunks = chunks

    async def chunks_for_project(
        self, _project_id: str, document_ids: tuple[str, ...] = ()
    ):
        """Return chunks with optional document filtering."""
        if not document_ids:
            return self.chunks
        return tuple(
            chunk for chunk in self.chunks if chunk.document_id in document_ids
        )

    async def chunks_by_ids(self, chunk_ids: tuple[str, ...]):
        """Hydrate selected chunks by ID."""
        return {
            chunk.chunk_id: chunk
            for chunk in self.chunks
            if chunk.chunk_id in chunk_ids
        }


@pytest.mark.asyncio
async def test_bm25_is_distinct_from_postgres_full_text_scoring() -> None:
    """BM25 ranks terms using corpus-local IDF and length normalization."""
    repository = _Repository(
        (
            _chunk("short", "vector search", 0),
            _chunk("long", "vector vector vector search unrelated terms", 1),
        )
    )
    response = await BM25Retriever(repository).retrieve(
        RetrievalRequest("project", "vector search"),
        RetrievalContext(repository=repository),
    )

    assert response.metadata["score_type"] == "bm25"
    assert response.results[0].retriever_id == "bm25"
    assert response.results[0].score is not None


def test_colbert_maxsim_sums_best_document_token_matches() -> None:
    """MaxSim chooses the best matching document token for every query token."""
    query = TokenEmbedding(((1.0, 0.0), (0.0, 1.0)), "colbert")
    document = TokenEmbedding(((1.0, 0.0), (0.5, 0.5)), "colbert")
    assert maxsim_score(query, document) == pytest.approx(1.5)


@dataclass
class _SparseEmbedder:
    """Small learned sparse encoder fixture."""

    model_identity: str = "bge-m3-test"

    async def embed_sparse_documents(self, texts: list[str]):
        return tuple(
            SparseEmbedding((index,), (1.0,), self.model_identity)
            for index, _text in enumerate(texts, 1)
        )

    async def embed_sparse_query(self, _text: str):
        return SparseEmbedding((1,), (1.0,), self.model_identity)


class _SparseIndex:
    """In-memory sparse index fixture with observable lazy indexing."""

    def __init__(self) -> None:
        self.vectors: tuple[tuple[RetrievalUnit, SparseEmbedding], ...] = ()
        self.upserts = 0

    async def has_project_model(self, _project_id: str) -> bool:
        return bool(self.vectors)

    async def upsert(self, _project_id: str, chunks, vectors) -> None:
        self.upserts += 1
        self.vectors = tuple(zip(chunks, vectors, strict=True))

    async def search(self, _project_id: str, query, top_k: int, _document_ids=()):
        ranked = [
            (
                sum(
                    left * right
                    for index, left in zip(vector.indices, vector.weights, strict=True)
                    for other, right in zip(query.indices, query.weights, strict=True)
                    if index == other
                ),
                chunk,
            )
            for chunk, vector in self.vectors
        ]
        ranked.sort(key=lambda item: (-item[0], item[1].chunk_id))
        return tuple(
            ChunkRetrievalResult(chunk, rank, "qdrant_sparse", value)
            for rank, (value, chunk) in enumerate(ranked[:top_k], 1)
        )


@pytest.mark.asyncio
async def test_bge_sparse_retriever_materializes_a_project_index_once() -> None:
    """Learned sparse retrieval persists document vectors and reuses them."""
    repository = _Repository((_chunk("one", "one", 0), _chunk("two", "two", 1)))
    index = _SparseIndex()
    retriever = BgeM3SparseRetriever(repository, _SparseEmbedder(), index)
    context = RetrievalContext(repository=repository)

    first = await retriever.retrieve(RetrievalRequest("project", "one"), context)
    second = await retriever.retrieve(RetrievalRequest("project", "one"), context)

    assert first.results[0].retriever_id == "bge_m3_sparse"
    assert second.results[0].retriever_id == "bge_m3_sparse"
    assert index.upserts == 1


@dataclass
class _ChildRetriever:
    """Static retriever fixture for weighted fusion."""

    results: tuple[ChunkRetrievalResult, ...]

    async def retrieve(self, request, _context):
        return RetrievalResponse(request.query, self.results)


@pytest.mark.asyncio
async def test_weighted_hybrid_normalizes_scores_before_fusing() -> None:
    """Weighted fusion keeps score scales from one child from dominating the other."""
    one = _chunk("one", "one", 0)
    two = _chunk("two", "two", 1)
    dense = _ChildRetriever(
        (
            ChunkRetrievalResult(one, 1, "dense", 100.0),
            ChunkRetrievalResult(two, 2, 50.0),
        )
    )
    sparse = _ChildRetriever(
        (ChunkRetrievalResult(two, 1, "sparse", 1.0), ChunkRetrievalResult(one, 2, 0.0))
    )

    response = await WeightedHybridRetriever(dense, sparse, top_k=2).retrieve(
        RetrievalRequest("project", "query"), RetrievalContext()
    )

    assert [result.chunk.chunk_id for result in response.results] == ["one", "two"]
    assert response.metadata["normalization"] == "min_max"
