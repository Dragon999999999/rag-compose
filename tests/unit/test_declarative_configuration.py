"""Verify standalone model/component registries and composition-only YAML."""

from pathlib import Path

import pytest
import yaml

from llm_grounding.infrastructure.config.declarative import DeclarativePipelineConfig
from llm_grounding.infrastructure.config.loader import (
    load_configuration_registry,
    load_model_catalog,
    load_pipeline_config,
)
from llm_grounding.infrastructure.config.registry import ConfigurationRegistry


def test_standalone_registry_resolves_ids_and_preserves_model_settings() -> None:
    """Different internal IDs retain their explicit stage and runtime settings."""
    registry = load_configuration_registry()

    first = registry.model("reasoning_001")
    assert first.stage == "reasoner"
    assert first.timeout_seconds > 0
    assert registry.component("reasoner", "reasoner_basic_grounded").model_id == (
        "reasoning_001"
    )


def test_composition_only_pipeline_has_no_provider_settings() -> None:
    """Pipeline source files contain only component IDs and composition metadata."""
    value = yaml.safe_load(
        Path("config/pipelines/query_basic_v1.yaml").read_text(encoding="utf-8")
    )

    assert value["query"]["retriever"] == "retriever_vector"
    assert value["query"]["reasoner"] == "reasoner_basic_grounded"
    assert "base_url" not in str(value)
    assert "model_name" not in str(value)
    assert "top_k" not in str(value)


def test_model_catalog_uses_explicit_stage_tags() -> None:
    """Standalone model stages determine catalog membership instead of backend type."""
    catalog = load_configuration_registry().model_catalog()

    assert "embedding_001" in catalog.embedding_models
    assert "embedding_001" not in catalog.reasoning_models
    assert "reasoning_001" in catalog.reasoning_models
    assert "reasoning_001" not in catalog.embedding_models
    assert "cross_encoder_minilm" not in catalog.embedding_models
    assert "cross_encoder_minilm" not in catalog.reasoning_models


def test_duplicate_standalone_ids_are_rejected(tmp_path: Path) -> None:
    """IDs, rather than filenames, are authoritative and must be unique."""
    models = tmp_path / "models"
    components = tmp_path / "components" / "parser"
    models.mkdir()
    components.mkdir(parents=True)
    model = """\
id: duplicate
stage: embedding
backend: sentence_transformers
name: sentence-transformers/all-MiniLM-L6-v2
"""
    (models / "first.yaml").write_text(model, encoding="utf-8")
    (models / "second.yaml").write_text(model, encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate model configuration ID: duplicate"):
        ConfigurationRegistry.from_directories(models, tmp_path / "components")


def test_missing_component_reference_is_rejected() -> None:
    """Composition cannot select a component that is not in the registry."""
    registry = load_configuration_registry()
    source = DeclarativePipelineConfig(
        kind="query",
        name="invalid",
        version=1,
        query={
            "retriever": "missing_retriever",
            "reranker": "reranker_identity",
            "reasoner": "reasoner_basic_grounded",
            "assessors": ("assessor_source_reference",),
            "decision_policy": "decision_basic_grounding",
            "renderer": "renderer_grounded_answer",
        },
    )

    with pytest.raises(
        ValueError, match="Unknown retriever component configuration ID"
    ):
        registry.pipeline(source)


def test_default_query_pipeline_resolves_from_component_ids() -> None:
    """The current query pipeline resolves reusable component definitions."""
    config = load_pipeline_config(Path("config/pipelines/query_basic_v1.yaml"))

    assert config.kind == "query"
    assert config.query is not None
    assert config.query.reasoner.type == "basic_grounded"
    assert config.query.reasoner.model_ref in load_model_catalog().reasoning_models
