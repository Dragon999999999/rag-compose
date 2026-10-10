"""Test deterministic canonical observation, stage, trace, and usage mapping."""

from datetime import UTC, datetime

from llm_grounding.domain.answers import (
    Answer,
    Assessment,
    DecisionAction,
    DecisionResult,
    ReasoningResult,
)
from llm_grounding.domain.corpus import (
    ChunkRetrievalResult,
    CorpusRequest,
    EvidenceReference,
    RetrievalUnit,
)
from llm_grounding.observability.builder import ObservationBuilder
from llm_grounding.observability.models import (
    LLMUsage,
    RetrievalStageType,
    UsageOrigin,
)
from llm_grounding.observability.trace import TraceRecorder
from llm_grounding.observability.usage import UsageAccumulator
from llm_grounding.orchestration.retrieval_stages import RetrievalStageBuilder
from llm_grounding.reasoning.basic_grounded import GroundedResponsePayload


def _result(chunk_id: str = "chunk-1") -> ChunkRetrievalResult:
    """Create one scored chunk with canonical evidence span metadata."""
    return ChunkRetrievalResult(
        RetrievalUnit(
            chunk_id,
            "document-1",
            "ingestion-1",
            "paper.pdf",
            "Supported source text.",
            0,
            2,
            2,
            (EvidenceReference("evidence-1", 4, 26),),
        ),
        1,
        "vector",
        0.91,
    )


def test_builder_preserves_claim_citation_and_stage_identity() -> None:
    """Canonical claims point to resolved source records in final context order."""
    result = _result()
    payload = GroundedResponsePayload(
        answer_text="Supported source text.",
        claims=[{"text": "Supported source text.", "source_chunk_ids": ["chunk-1"]}],
        insufficient_evidence=False,
    )
    reasoning = ReasoningResult("run-1", payload.model_dump(mode="json"))
    observation = ObservationBuilder().build(
        run_id="run-1",
        request_id="request-1",
        corpus_id="corpus-1",
        architecture_name="basic_rag_v1",
        architecture_version=1,
        request=CorpusRequest("corpus-1", "What is supported?"),
        chunks=(result,),
        answer=Answer("Supported source text.", payload=payload.model_dump()),
        reasoning=reasoning,
        assessments=(
            Assessment("run-1", "source_reference", "source_reference", 1.0, "pass"),
        ),
        decision=DecisionResult(
            DecisionAction.ACCEPT, reasoning, {"reason": "validated"}
        ),
        created_at=datetime.now(UTC),
    )

    assert (
        observation.retrieval[0].stage_type
        == RetrievalStageType.CANDIDATE_RETRIEVAL
    )
    assert observation.retrieval[1].parent_stage_ids == [
        observation.retrieval[0].stage_id
    ]
    assert observation.answer is not None
    assert observation.answer.citations[0].resolved is True
    assert observation.answer.citations[0].start_char == 4
    assert observation.decision is not None
    assert observation.decision.action == "accept"


def test_stage_builder_distinguishes_empty_output_and_removed_items() -> None:
    """Evaluation can distinguish an empty final context from a missing stage."""
    result = _result()
    builder = RetrievalStageBuilder(run_id="run-1")
    candidate = builder.from_results(RetrievalStageType.CANDIDATE_RETRIEVAL, (result,))
    final = builder.build(
        RetrievalStageType.FINAL_CONTEXT,
        [],
        parent_stage_ids=(candidate.stage_id,),
        removed_item_ids=(result.chunk.chunk_id,),
    )

    assert final.status == "empty"
    assert final.parent_stage_ids == [candidate.stage_id]
    assert final.removed_item_ids == [result.chunk.chunk_id]


def test_trace_and_usage_retain_nested_relationships_without_fabrication() -> None:
    """Timing and provider usage are aggregated while unknown values stay null."""
    traces = TraceRecorder()
    parent = traces.start("query")
    child = traces.start("retrieval")
    traces.end(child, output_count=1)
    traces.end(parent, output_count=1)
    usage = UsageAccumulator()
    usage.add_llm(
        LLMUsage(
            model_id="model",
            input_tokens=10,
            output_tokens=4,
            total_tokens=14,
            usage_origin=UsageOrigin.PROVIDER_REPORTED,
        )
    )

    assert traces.spans[1].parent_span_id == traces.spans[0].span_id
    assert all(span.duration_ms is not None for span in traces.spans)
    assert usage.build().total_tokens == 14
    assert usage.build().context_tokens is None
