"""Verify deterministic assessor contracts and native score preservation."""

from llm_grounding.assessment.deterministic import (
    DiversityAssessor,
    EvidenceSufficiencyAssessor,
    NoOpAssessor,
    RelevanceAssessor,
    RetrievalConfidenceAssessor,
)
from llm_grounding.domain.answers import ReasoningResult
from llm_grounding.domain.corpus import (
    ChunkRetrievalResult,
    CorpusRequest,
    EvidenceReference,
    GroundingContext,
    RetrievalUnit,
)
from llm_grounding.domain.evidence import RankingScore


def _context() -> GroundingContext:
    """Build a small score-bearing context for assessor tests."""
    first = RetrievalUnit(
        "chunk-1",
        "document",
        "ingestion",
        "paper.pdf",
        "The protocol uses photons and has a method section.",
        0,
        1,
        1,
        (EvidenceReference("evidence-1", 0, 50),),
        {"entities": ["protocol"]},
    )
    second = RetrievalUnit(
        "chunk-2",
        "document",
        "ingestion",
        "paper.pdf",
        "The protocol reports results in the conclusion.",
        1,
        1,
        1,
        (EvidenceReference("evidence-2", 0, 47),),
    )
    return GroundingContext(
        CorpusRequest("project", "What method does the protocol use?"),
        (
            ChunkRetrievalResult(
                first,
                1,
                "bm25",
                3.0,
                RankingScore(3.0, "bm25", "bm25"),
                RankingScore(0.8, "cross_encoder_logit", "reranker"),
            ),
            ChunkRetrievalResult(second, 2, "bm25", 1.0),
        ),
    )


async def test_noop_assessor_does_not_process_context() -> None:
    """No-op assessment returns no measurements and no resource work."""
    assert (
        await NoOpAssessor().assess_context(ReasoningResult("run", {}), _context())
        == ()
    )


async def test_confidence_reports_native_statistics_and_threshold() -> None:
    """Confidence retains score kind and reports deterministic statistics."""
    assessment = (
        await RetrievalConfidenceAssessor(
            threshold=1.5, score_kind="bm25", min_threshold_count=1
        ).assess_context(ReasoningResult("run", {}), _context())
    )[0]

    assert assessment.status == "passed"
    assert assessment.metadata["score_kind"] == "bm25"
    assert assessment.metadata["maximum"] == 3.0


async def test_relevance_uses_rerank_score_without_overwriting_retrieval_score(
) -> None:
    """Relevance reads the separate rerank score layer."""
    assessment = (
        await RelevanceAssessor(threshold=0.5).assess_context(
            ReasoningResult("run", {}), _context()
        )
    )[0]

    assert assessment.status == "passed"
    assert assessment.score == 0.8
    assert _context().chunks[0].ranking_score is not None
    assert _context().chunks[0].ranking_score.value == 3.0


async def test_sufficiency_and_diversity_are_heuristic_measurements() -> None:
    """Coverage and redundancy expose transparent deterministic metadata."""
    context = _context()
    result = ReasoningResult("run", {})
    sufficiency = (
        await EvidenceSufficiencyAssessor(
            requested_aspects=("method", "results")
        ).assess_context(result, context)
    )[0]
    diversity = (await DiversityAssessor().assess_context(result, context))[0]

    assert sufficiency.metadata["heuristic"] is True
    assert "method" in sufficiency.metadata["covered_aspects"]
    assert diversity.metadata["pair_count"] == 1
