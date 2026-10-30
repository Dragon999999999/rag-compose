"""Verify readiness diagnostics for selected ingestion and query stages."""

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_grounding.application.pipeline_readiness import (
    log_ingestion_pipeline_ready,
    log_query_pipeline_ready,
)
from llm_grounding.infrastructure.config.loader import (
    load_model_catalog,
    load_pipeline_config,
)
from llm_grounding.ingestion.models import IngestionConfig


def test_ingestion_readiness_logs_each_configured_stage(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Project creation diagnostics identify every effective ingestion stage."""
    caplog.set_level(logging.INFO, logger="llm_grounding.pipeline_readiness")

    log_ingestion_pipeline_ready(
        SimpleNamespace(kind="ingestion", name="ingestion", id="pipeline-id"),
        IngestionConfig(),  # type: ignore[arg-type]
    )

    messages = [record.getMessage() for record in caplog.records]
    assert any("Stage parser pymupdf is ready" in message for message in messages)
    assert any(
        "Stage knowledge builder basic is ready" in message for message in messages
    )
    assert any("Stage chunker block_aware is ready" in message for message in messages)
    assert any(
        "Stage embedder sentence-transformers/all-MiniLM-L6-v2 is ready" in message
        for message in messages
    )
    assert any(
        "Stage vector backend qdrant is ready" in message for message in messages
    )


def test_query_readiness_logs_each_configured_stage(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Chat creation diagnostics identify every effective query stage."""
    caplog.set_level(logging.INFO, logger="llm_grounding.pipeline_readiness")
    config = load_pipeline_config(
        Path("config/pipelines/query_basic_v1.yaml"),
        catalog=load_model_catalog(),
    )
    pipeline = SimpleNamespace(
        kind="query",
        name=config.name,
        id="pipeline-id",
        config=config.model_dump(mode="json"),
    )

    log_query_pipeline_ready(pipeline)  # type: ignore[arg-type]

    messages = [record.getMessage() for record in caplog.records]
    for stage in ("retriever", "reranker", "decision policy", "reasoner", "renderer"):
        assert any(f"Stage {stage} " in message for message in messages)
    assert any("Stage assessor " in message for message in messages)
