"""Verify opt-in structured grounding diagnostics."""

import json
import logging

import pytest

from llm_grounding.domain.answers import (
    Assessment,
    DecisionAction,
    DecisionResult,
    ReasoningResult,
)
from llm_grounding.domain.corpus import (
    ChunkRetrievalResult,
    CorpusRequest,
    EvidenceReference,
    GroundingContext,
    RetrievalUnit,
)
from llm_grounding.orchestration.grounding_diagnostics import (
    GroundingDiagnostics,
    GroundingDiagnosticsLevel,
)


def _grounding_inputs() -> tuple[
    GroundingContext, ReasoningResult, tuple[Assessment, ...], DecisionResult
]:
    """Create one canonicalized result and its corresponding raw model response."""
    chunk_id = "ingestion:c000016"
    chunk = RetrievalUnit(
        chunk_id,
        "document",
        "ingestion",
        "paper.pdf",
        "The paper introduces a framework.",
        0,
        2,
        2,
        (EvidenceReference("evidence", 0, 33),),
    )
    context = GroundingContext(
        CorpusRequest("project", "What is the contribution?"),
        (ChunkRetrievalResult(chunk, 1, "qdrant", 0.9),),
    )
    raw_payload = {
        "answer_text": "The paper introduces a framework.",
        "claims": [
            {
                "text": "The paper introduces a framework.",
                "source_chunk_ids": ["c000016"],
            }
        ],
        "insufficient_evidence": False,
        "limitations": None,
    }
    canonical_payload = {
        **raw_payload,
        "claims": [
            {
                "text": "The paper introduces a framework.",
                "source_chunk_ids": [chunk_id],
            }
        ],
    }
    reasoning = ReasoningResult(
        "result",
        canonical_payload,
        metadata={"raw_model_payload": raw_payload},
    )
    assessments = (
        Assessment(
            "result",
            "source_reference",
            "source_reference",
            1.0,
            "pass",
            (chunk_id,),
            {"unknown_chunk_ids": []},
        ),
    )
    decision = DecisionResult(DecisionAction.ACCEPT, reasoning)
    return context, reasoning, assessments, decision


def _record_from(caplog: pytest.LogCaptureFixture) -> dict[str, object]:
    """Parse the one JSON diagnostic record captured for a test."""
    message = caplog.records[-1].getMessage()
    return json.loads(message.removeprefix("Grounding diagnostics:\n"))


def test_verbose_diagnostics_include_raw_response_and_grounding_inputs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verbose output retains model output and all inputs used for validation."""
    context, reasoning, assessments, decision = _grounding_inputs()
    caplog.set_level(logging.INFO, logger="llm_grounding.grounding_diagnostics")

    GroundingDiagnostics(GroundingDiagnosticsLevel.VERBOSE).report(
        context, reasoning, assessments, decision
    )

    record = _record_from(caplog)
    assert record["llm"] == {
        "raw_response": reasoning.metadata["raw_model_payload"],
        "canonicalized_response": reasoning.payload,
        "reasoning_metadata": reasoning.metadata,
    }
    assert record["context"] == {
        "history": [],
        "chunks": [
            {
                "id": "ingestion:c000016",
                "document_id": "document",
                "document_name": "paper.pdf",
                "page_start": 2,
                "page_end": 2,
                "text": "The paper introduces a framework.",
                "evidence_refs": [{"evidence_id": "evidence", "start": 0, "end": 33}],
                "metadata": {},
            }
        ],
    }
    assert record["assessments"][0]["label"] == "pass"
    assert record["decision"]["action"] == "accept"


def test_summary_diagnostics_exclude_raw_response_and_document_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Summary output remains suitable for lower-sensitivity monitoring."""
    context, reasoning, assessments, decision = _grounding_inputs()
    caplog.set_level(logging.INFO, logger="llm_grounding.grounding_diagnostics")

    GroundingDiagnostics(GroundingDiagnosticsLevel.SUMMARY).report(
        context, reasoning, assessments, decision
    )

    record = _record_from(caplog)
    assert "llm" not in record
    assert "context" not in record
    assert record["retrieval"] == {
        "chunk_count": 1,
        "chunks": [
            {
                "id": "ingestion:c000016",
                "rank": 1,
                "score": 0.9,
                "retriever_id": "qdrant",
            }
        ],
    }
