"""Round-trip coverage for optional canonical observation fields."""

from datetime import UTC, datetime

from llm_grounding.observability.models import (
    Confidence,
    ConversationState,
    EmbeddingDiagnostics,
    GenerationInfo,
    RagRunObservation,
    ResourceUsage,
    StageSnapshot,
)


def test_observation_round_trips_advanced_optional_fields() -> None:
    """Persistence JSON retains semantic, experiment, DAG, and telemetry data."""
    timestamp = datetime.now(UTC)
    observation = RagRunObservation(
        run_id="run",
        request_id="request",
        target_run_id="target-run",
        request={"query": "question"},
        corpus_id="eval-corpus",
        architecture_name="query_basic_v1",
        architecture_version=1,
        status="abstained",
        generation=GenerationInfo(
            model_id="model-v2", prompt_id="prompt", prompt_version="3"
        ),
        stages=[
            StageSnapshot(
                stage_id="stage-1",
                stage_type="custom_rewrite",
                component_name="rewriter",
                output_item_ids=["query-1"],
                metadata={"branch": "a"},
            )
        ],
        embedding=EmbeddingDiagnostics(
            embedding_id="embedding",
            model_id="embed-v1",
            dimension=3,
            vector_norm=1.0,
        ),
        trace=[
            {
                "span_id": "span",
                "name": "query",
                "started_at": timestamp,
                "resource_usage": ResourceUsage(
                    component_id="query",
                    span_id="span",
                    measurement_scope="process",
                    measurement_source="stdlib",
                ),
            }
        ],
        conversation_state=ConversationState(
            prior_turn_ids=["turn-0"], included_history_turn_ids=["turn-0"]
        ),
        answer={
            "status": "abstained",
            "generation_status": "abstained",
            "confidence": Confidence(),
        },
        experiment_id="experiment",
        benchmark_case_id="case",
        seed=7,
        configuration_hash="config-hash",
        created_at=timestamp,
        completed_at=timestamp,
    )

    restored = RagRunObservation.model_validate(observation.model_dump(mode="json"))

    assert restored == observation
    assert restored.trace[0].resource_usage is not None
    assert restored.stages[0].output_item_ids == ["query-1"]
