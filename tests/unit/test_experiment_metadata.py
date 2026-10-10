"""Deterministic experiment identity and status metadata tests."""

from datetime import UTC, datetime

from llm_grounding.application.eval_corpus_service import EvalCorpusService
from llm_grounding.observability.models import RagRunObservation


def test_deterministic_request_identity_preserves_case_and_experiment() -> None:
    """Different benchmark cases or experiments cannot alias one request."""
    first = EvalCorpusService.deterministic_request_id(
        corpus_id="eval",
        benchmark_case_id="case-a",
        experiment_id="experiment-a",
        query="same question",
    )
    second = EvalCorpusService.deterministic_request_id(
        corpus_id="eval",
        benchmark_case_id="case-b",
        experiment_id="experiment-a",
        query="same question",
    )

    assert first == EvalCorpusService.deterministic_request_id(
        corpus_id="eval",
        benchmark_case_id="case-a",
        experiment_id="experiment-a",
        query="same question",
    )
    assert first != second


def test_observation_records_model_config_time_and_distinct_status() -> None:
    """A persisted observation keeps model/config identity and lifecycle state."""
    now = datetime.now(UTC)
    observation = RagRunObservation(
        run_id="run",
        request_id="request",
        corpus_id="corpus",
        architecture_name="basic_rag_v1",
        architecture_version=1,
        status="timeout",
        request={"query": "question"},
        generation={"model_id": "target-model"},
        configuration_hash="config-hash",
        experiment_id="experiment",
        benchmark_case_id="case",
        seed=11,
        created_at=now,
        completed_at=now,
        error="TimeoutError",
    )

    assert observation.status == "timeout"
    assert observation.generation.model_id == "target-model"
    assert observation.completed_at == now
    assert observation.error == "TimeoutError"
