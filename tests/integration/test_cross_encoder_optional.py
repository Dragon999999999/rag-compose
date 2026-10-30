"""Optional smoke test for locally cached MiniLM cross-encoder weights."""

import os

import pytest

from llm_grounding.domain.evidence import EvidenceType, EvidenceUnit, RetrievalResult
from llm_grounding.retrieval.models import RetrievalResponse
from llm_grounding.retrieval.reranking.cross_encoder import CrossEncoderReranker


def _response() -> RetrievalResponse:
    """Create a small query/candidate pair set for the optional model test."""
    return RetrievalResponse(
        "What is the capital of France?",
        (
            RetrievalResult(
                EvidenceUnit(
                    "relevant",
                    "doc",
                    EvidenceType.TEXT,
                    "",
                    0,
                    text="Paris is the capital of France.",
                ),
                1,
                "vector",
            ),
            RetrievalResult(
                EvidenceUnit(
                    "irrelevant",
                    "doc",
                    EvidenceType.TEXT,
                    "",
                    1,
                    text="Berlin is the capital of Germany.",
                ),
                2,
                "vector",
            ),
        ),
    )


@pytest.mark.skipif(
    os.environ.get("RUN_LOCAL_RERANKER_TESTS") != "1",
    reason="Set RUN_LOCAL_RERANKER_TESTS=1 to load the optional MiniLM model",
)
async def test_local_minilm_cross_encoder() -> None:
    """Run MiniLM only when explicitly requested by a developer or CI job."""
    response = await CrossEncoderReranker(top_k=1).rerank(_response())

    assert response.results[0].evidence.evidence_id == "relevant"
