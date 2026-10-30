"""Verify strategy-level retrieval contracts without external services."""

from dataclasses import dataclass

import pytest

from llm_grounding.domain.corpus import ChunkRetrievalResult, RetrievalUnit
from llm_grounding.retrieval.full_context import FullContextRetriever
from llm_grounding.retrieval.hybrid import HybridRetriever
from llm_grounding.retrieval.models import (
    RetrievalContext,
    RetrievalRequest,
    RetrievalResponse,
)
from llm_grounding.retrieval.sparse import SparseRetriever


def _chunk(chunk_id: str, document_id: str, order: int, text: str) -> RetrievalUnit:
    """Create a provenance-bearing test chunk."""
    return RetrievalUnit(
        chunk_id,
        document_id,
        "ingestion",
        f"{document_id}.pdf",
        text,
        order,
        1,
        1,
        (),
    )


@dataclass
class _ChunkRepository:
    """Supply canonical chunks to retrieval unit tests."""

    chunks: tuple[RetrievalUnit, ...]

    async def chunks_for_project(
        self, project_id: str, document_ids: tuple[str, ...] = ()
    ) -> tuple[RetrievalUnit, ...]:
        """Apply the same project/document-filtered contract as PostgreSQL."""
        del project_id
        return tuple(
            chunk
            for chunk in self.chunks
            if not document_ids or chunk.document_id in document_ids
        )


@pytest.mark.asyncio
async def test_full_context_reports_budget_omissions_without_embedding_dependencies(
) -> None:
    """Full-context retrieval preserves order and reports omitted chunks."""
    repository = _ChunkRepository(
        (
            _chunk("a", "doc-a", 1, "a" * 20),
            _chunk("b", "doc-a", 2, "b" * 20),
        )
    )
    retriever = FullContextRetriever(repository, context_budget_tokens=6)

    response = await retriever.retrieve(
        RetrievalRequest("project", "ignored"), RetrievalContext()
    )

    assert [item.chunk.chunk_id for item in response.results] == ["a"]
    assert response.metadata["budget_exceeded"] is True
    assert response.metadata["omitted_chunk_ids"] == ["b"]


@pytest.mark.asyncio
async def test_sparse_retriever_uses_repository_fts_contract() -> None:
    """Sparse retrieval returns the repository's ranked canonical results."""
    chunk = _chunk("s", "doc", 1, "matching")

    class Repository(_ChunkRepository):
        async def search_chunks_fts(self, project_id, query, top_k, document_ids=()):
            assert (project_id, query, top_k, document_ids) == (
                "project",
                "term",
                4,
                (),
            )
            return (ChunkRetrievalResult(chunk, 1, "sparse", 0.75),)

    response = await SparseRetriever(Repository((chunk,)), top_k=4).retrieve(
        RetrievalRequest("project", "term"), RetrievalContext()
    )

    assert response.results[0].score == 0.75
    assert response.metadata["score_type"] == "postgres_ts_rank_cd"


@pytest.mark.asyncio
async def test_hybrid_retriever_deduplicates_and_preserves_component_metadata() -> None:
    """Hybrid fusion uses stable chunk IDs and the standard weighted RRF formula."""
    first = _chunk("first", "doc", 1, "first")
    second = _chunk("second", "doc", 2, "second")

    class Retriever:
        def __init__(self, results):
            self.results = results

        async def retrieve(self, request, context):
            del request, context
            return RetrievalResponse("term", tuple(self.results))

    hybrid = HybridRetriever(
        Retriever(
            [
                ChunkRetrievalResult(first, 1, "dense", 0.9),
                ChunkRetrievalResult(second, 2, "dense", 0.8),
            ]
        ),
        Retriever(
            [
                ChunkRetrievalResult(second, 1, "sparse", 0.7),
            ]
        ),
        dense_top_k=5,
        sparse_top_k=5,
        top_k=2,
        rrf_k=1,
    )

    response = await hybrid.retrieve(
        RetrievalRequest("project", "term"), RetrievalContext()
    )

    assert [item.chunk.chunk_id for item in response.results] == ["second", "first"]
    assert response.results[0].score == pytest.approx(1 / 3 + 1 / 2)
    assert response.results[1].score == pytest.approx(1 / 2)
    assert response.metadata["component_scores"]["second"] == {
        "dense_score": 0.8,
        "dense_rank": 2,
        "sparse_score": 0.7,
        "sparse_rank": 1,
    }
