"""Exercise prompt preparation, output constraints, and policy/rendering boundaries."""

from pathlib import Path

import pytest
from conftest import RecordingModelClient
from pydantic import ValidationError

from llm_grounding.assessment.noop import NoOpAssessor
from llm_grounding.decision.accept_all import AcceptAllDecisionPolicy
from llm_grounding.domain.answers import DecisionAction, DecisionResult, ReasoningResult
from llm_grounding.domain.documents import DocumentSource
from llm_grounding.knowledge.basic import BasicKnowledgeBuilder
from llm_grounding.parsing.pymupdf import PyMuPDFDocumentParser
from llm_grounding.reasoning.basic_summary import (
    BasicSummaryPayload,
    BasicSummaryReasoner,
)
from llm_grounding.reasoning.models import ReasoningRequest
from llm_grounding.rendering.basic_summary import BasicSummaryRenderer
from llm_grounding.retrieval.full_document import FullDocumentRetriever
from llm_grounding.retrieval.models import RetrievalRequest


async def test_reasoner_orders_full_text_and_propagates_validated_payload(
    pdf_path: Path, model_client: RecordingModelClient
) -> None:
    """Scrambled candidates are concatenated in source order at reasoning time."""
    parsed = await PyMuPDFDocumentParser().parse(DocumentSource("paper", pdf_path))
    knowledge = await BasicKnowledgeBuilder().build(parsed)
    retrieved = await FullDocumentRetriever().retrieve(
        RetrievalRequest("summarize", knowledge)
    )
    request = ReasoningRequest("summarize", tuple(reversed(retrieved.results)))
    result = await BasicSummaryReasoner(model_client).reason(request)
    assert result.payload == model_client.payload.model_dump()
    assert result.claims == ()
    assert result.metadata["prompt_version"] == "1.0"
    prompt = model_client.prompts[0]
    assert (
        prompt.index("Upper block")
        < prompt.index("Lower block")
        < prompt.index("Second page")
    )
    assert "5-8 keywords" in prompt
    assert "2-3 sentence summary" in prompt
    assert "User request: summarize" in prompt


async def test_reasoner_rejects_empty_evidence_before_model_call(
    model_client: RecordingModelClient,
) -> None:
    """Textless requests fail locally and cannot generate unsupported summaries."""
    with pytest.raises(ValueError, match="nonempty textual evidence"):
        await BasicSummaryReasoner(model_client).reason(
            ReasoningRequest("summarize", ())
        )
    assert model_client.prompts == []


@pytest.mark.parametrize("count,length", [(5, 20), (8, 500)])
def test_output_schema_accepts_original_boundaries(count: int, length: int) -> None:
    """Inclusive keyword and summary limits match the original extraction script."""
    assert BasicSummaryPayload(keywords=["keyword"] * count, summary="x" * length)


@pytest.mark.parametrize("count,length", [(4, 20), (9, 20), (5, 19), (5, 501)])
def test_output_schema_rejects_out_of_range_values(count: int, length: int) -> None:
    """Invalid structured model output cannot pass the baseline schema."""
    with pytest.raises(ValidationError):
        BasicSummaryPayload(keywords=["keyword"] * count, summary="x" * length)


async def test_assessment_decision_and_rendering_are_distinct(
    payload: BasicSummaryPayload,
) -> None:
    """No-op assessment makes no judgment, while policy explicitly authorizes output."""
    reasoning = ReasoningResult("result", payload.model_dump())
    assessments = await NoOpAssessor().assess(reasoning, ())
    assert assessments == ()
    decision = await AcceptAllDecisionPolicy().decide(reasoning, assessments)
    assert decision.reasoning is reasoning
    assert decision.action == DecisionAction.ACCEPT
    renderer = BasicSummaryRenderer()
    answer = await renderer.render(decision)
    assert answer == await renderer.render(decision)
    assert answer.payload == reasoning.payload
    assert answer.text.startswith("Keywords:\n1. grounding")
    assert "Summary:\n" in answer.text
    assert answer.claims == ()
    assert answer.evidence_ids == ()
    assert answer.metadata["verified"] is False


@pytest.mark.parametrize("action", [DecisionAction.REJECT, DecisionAction.ABSTAIN])
async def test_renderer_never_authorizes_rejected_output(
    payload: BasicSummaryPayload, action: DecisionAction
) -> None:
    """The renderer respects policy instead of independently accepting content."""
    decision = DecisionResult(action, ReasoningResult("result", payload.model_dump()))
    with pytest.raises(ValueError, match="accepted decision"):
        await BasicSummaryRenderer().render(decision)


async def test_renderer_rejects_incompatible_payload() -> None:
    """Architecture-specific renderers validate the structured payload they consume."""
    decision = DecisionResult(DecisionAction.ACCEPT, ReasoningResult("result", {}))
    with pytest.raises(ValidationError):
        await BasicSummaryRenderer().render(decision)
