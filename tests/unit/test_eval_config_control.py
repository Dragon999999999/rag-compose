"""Unit tests for request-local evaluation configuration control."""

import pytest

from llm_grounding.application.eval_config import EvaluationConfigService
from llm_grounding.ingestion.models import IngestionConfig


def test_overrides_are_validated_and_recorded_without_mutating_base_config(
    config,
) -> None:
    """An effective configuration contains overrides while the base stays unchanged."""
    architecture = config.model_copy(
        update={
            "name": "basic_rag_v1",
            "retriever": config.retriever.model_copy(update={"type": "vector"}),
        }
    )
    ingestion = IngestionConfig()

    result = EvaluationConfigService().validate(
        architecture,
        ingestion,
        {"top_k": 5, "prompt_id": "grounded_qa"},
        corpus_id="eval-1",
    )

    assert result.effective.retrieval is not None
    assert result.effective.retrieval.top_k == 5
    assert result.effective.prompt is not None
    assert result.effective.prompt.prompt_id == "grounded_qa"
    assert architecture.retriever.top_k != 5
    assert architecture.reasoner.prompt_id != "grounded_qa"


def test_unsupported_override_is_rejected(config) -> None:
    """Advanced stages cannot be silently claimed by basic_rag_v1."""
    architecture = config.model_copy(
        update={
            "name": "basic_rag_v1",
            "retriever": config.retriever.model_copy(update={"type": "vector"}),
        }
    )

    with pytest.raises(ValueError, match="Unsupported evaluation configuration"):
        EvaluationConfigService().validate(
            architecture, IngestionConfig(), {"reranker": "cross_encoder"}
        )
