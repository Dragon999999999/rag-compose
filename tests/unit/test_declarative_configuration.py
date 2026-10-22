"""Verify standalone model/component registries and composition-only YAML."""

from pathlib import Path

import pytest
import yaml

from llm_grounding.infrastructure.config.declarative import (
    DeclarativeArchitectureConfig,
)
from llm_grounding.infrastructure.config.loader import (
    load_architecture_config,
    load_configuration_registry,
)
from llm_grounding.infrastructure.config.registry import ConfigurationRegistry


def test_standalone_registry_resolves_ids_and_preserves_model_settings() -> None:
    """Different internal IDs retain their explicit stage and runtime settings."""
    registry = load_configuration_registry()

    first = registry.model("reasoning_001")
    second = registry.model("reasoning_002")

    assert first.id != second.id
    assert first.stage == second.stage == "reasoner"
    assert first.timeout_seconds != second.timeout_seconds
    assert registry.component("reasoner", "reasoner_basic_summary").model_id == (
        "reasoning_001"
    )


def test_composition_only_pipeline_has_no_provider_settings() -> None:
    """Pipeline source files contain only component IDs and composition metadata."""
    value = yaml.safe_load(
        Path("config/pipelines/basic_rag_v1.yaml").read_text(encoding="utf-8")
    )

    assert value["representation"]["parser"] == "parser_pymupdf"
    assert value["query"]["reasoner"] == "reasoner_basic_grounded"
    assert "base_url" not in str(value)
    assert "model_name" not in str(value)
    assert "top_k" not in str(value)


def test_model_catalog_uses_explicit_stage_tags() -> None:
    """Standalone model stages determine catalog membership instead of backend type."""
    catalog = load_configuration_registry().model_catalog()

    assert "aqueduct_qwen_embedding" in catalog.embedding_models
    assert "aqueduct_qwen_embedding" not in catalog.reasoning_models
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
    source = DeclarativeArchitectureConfig(
        name="invalid",
        version=1,
        parser="missing_parser",
        knowledge_builder="knowledge_basic",
        retriever="retriever_full_document",
        reranker="reranker_identity",
        reasoner="reasoner_basic_summary",
        assessors=("assessor_noop",),
        decision_policy="decision_accept_all",
        renderer="renderer_basic_summary",
    )

    with pytest.raises(ValueError, match="Unknown parser component configuration ID"):
        registry.architecture(source)


def test_default_architecture_is_resolved_from_component_ids() -> None:
    """The application default now resolves reusable component definitions."""
    registry = load_configuration_registry()
    config = load_architecture_config()

    assert config.parser.type == "pymupdf"
    assert config.reasoner.model.name == registry.model("reasoning_001").name
    assert config.embedder is not None
    assert config.embedder.model.name == "qwen3-embedding-4b"
