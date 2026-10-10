"""Acceptance checks for request-local configuration control."""

from llm_grounding.application.eval_config import EvaluationConfigService
from llm_grounding.ingestion.models import IngestionConfig


def test_config_override_is_recorded_in_effective_hash(config) -> None:
    """Two effective top-k settings produce different reproducibility identities."""
    architecture = config.model_copy(
        update={
            "name": "basic_rag_v1",
            "retriever": config.retriever.model_copy(update={"type": "vector"}),
        }
    )
    service = EvaluationConfigService()
    first = service.validate(architecture, IngestionConfig(), {"top_k": 3})
    second = service.validate(architecture, IngestionConfig(), {"top_k": 4})

    assert first.effective.retrieval.top_k == 3
    assert second.effective.retrieval.top_k == 4
    assert (
        first.effective.architecture.config_hash
        != second.effective.architecture.config_hash
    )
