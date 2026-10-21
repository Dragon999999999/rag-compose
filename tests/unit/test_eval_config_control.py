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
            architecture, IngestionConfig(), {"fusion": "cross_encoder"}
        )


def test_reranking_overrides_are_validated_and_recorded(config) -> None:
    """Evaluation requests can select top-k or local cross-encoder settings."""
    architecture = config.model_copy(
        update={
            "name": "basic_rag_v1",
            "retriever": config.retriever.model_copy(update={"type": "vector"}),
        }
    )

    result = EvaluationConfigService().validate(
        architecture,
        IngestionConfig(),
        {
            "reranker": "cross_encoder",
            "retrieval_top_k": 20,
            "rerank_top_k": 5,
            "reranker_device": "cpu",
        },
    )

    assert result.effective.retrieval.reranker == "cross_encoder"
    assert result.effective.retrieval.retrieval_top_k == 20
    assert result.effective.retrieval.rerank_top_k == 5

    with pytest.raises(ValueError, match="must not exceed"):
        EvaluationConfigService().validate(
            architecture,
            IngestionConfig(),
            {"reranker": "top_k", "retrieval_top_k": 2, "rerank_top_k": 3},
        )
