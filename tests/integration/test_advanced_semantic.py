"""Acceptance checks for semantic-judge input completeness."""

from llm_grounding.observability.models import (
    CitationModel,
    GenerationInfo,
    ObservationAnswer,
    RagRunObservation,
)


def test_semantic_inputs_include_generation_history_and_source_evidence() -> None:
    """A judge can identify the generation and inspect cited source text."""
    observation = RagRunObservation(
        run_id="run",
        request_id="request",
        corpus_id="corpus",
        architecture_name="query_basic_v1",
        architecture_version=1,
        status="completed",
        request={"query": "question", "conversation_history": [{"id": "turn-0"}]},
        generation=GenerationInfo(
            model_id="model", prompt_id="prompt", prompt_version="1"
        ),
        answer=ObservationAnswer(
            text="answer",
            citations=[
                CitationModel(
                    citation_id="citation",
                    claim_id="claim",
                    chunk_id="chunk",
                    source_text="source text",
                    source_evidence_id="evidence",
                    resolved=True,
                )
            ],
        ),
        created_at="2026-01-01T00:00:00Z",
    )

    assert observation.generation.model_id == "model"
    assert observation.request.conversation_history[0]["id"] == "turn-0"
    assert observation.answer.citations[0].source_evidence_id == "evidence"
