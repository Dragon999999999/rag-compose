"""Acceptance checks for reproducible multi-turn observation state."""

from llm_grounding.observability.models import ConversationState, RagRunObservation


def test_multiturn_observation_keeps_exact_history_window() -> None:
    """The turn and included-history identities are persisted independently."""
    observation = RagRunObservation(
        run_id="run",
        request_id="request",
        corpus_id="corpus",
        architecture_name="query_basic_v1",
        architecture_version=1,
        status="completed",
        request={
            "query": "follow-up",
            "conversation_history": [{"id": "turn-1"}],
            "included_history_turn_ids": ["turn-1"],
        },
        conversation_id="conversation",
        turn_id="turn-2",
        turn_index=1,
        conversation_state=ConversationState(
            prior_turn_ids=["turn-1"], included_history_turn_ids=["turn-1"]
        ),
        created_at="2026-01-01T00:00:00Z",
    )

    assert observation.turn_id == "turn-2"
    assert observation.conversation_state.included_history_turn_ids == ["turn-1"]
