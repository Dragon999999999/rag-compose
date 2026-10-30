"""Validation coverage for the reproducible test pipeline presets."""

from pathlib import Path

import pytest

from llm_grounding.infrastructure.config.loader import load_pipeline_config

_TEST_PIPELINES = Path("config/pipelines/test")


@pytest.mark.parametrize(
    "filename",
    (
        "test_ingestion.yaml",
        "test_ingestion_baseline.yaml",
        "test_baseline.yaml",
        "test_1.yaml",
        "test_2.yaml",
        "test_3.yaml",
    ),
)
def test_test_pipeline_preset_is_registered_and_valid(filename: str) -> None:
    """Every test preset resolves through the closed component registry."""
    pipeline = load_pipeline_config(_TEST_PIPELINES / filename)

    assert pipeline.name == Path(filename).stem
