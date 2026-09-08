"""Verify typed configuration and identifier resolution before any network use."""

from pathlib import Path

import pytest

from llm_grounding.infrastructure.config.loader import (
    ConfigurationError,
    load_architecture_config,
)
from llm_grounding.infrastructure.config.models import ArchitectureConfig
from llm_grounding.orchestration.registry import ComponentRegistry


def test_default_configuration_matches_repository_yaml(
    config: ArchitectureConfig,
) -> None:
    """The application default is the same composition as the checked-in YAML."""
    assert load_architecture_config() == config
    assert config.name == "baseline_v0"
    assert config.version == 0
    assert config.reasoner.model.name == "qwen-3.5-397b"
    assert (
        str(config.reasoner.model.base_url)
        == "https://aqueduct.ai.datalab.tuwien.ac.at/v1"
    )


def test_registry_constructs_all_expected_components(
    config: ArchitectureConfig,
) -> None:
    """Construction resolves all stage identifiers without requiring credentials."""
    architecture = ComponentRegistry().build(config)
    expected = {
        "parser": "PyMuPDFDocumentParser",
        "knowledge_builder": "BasicKnowledgeBuilder",
        "retriever": "FullDocumentRetriever",
        "reranker": "IdentityReranker",
        "reasoner": "BasicSummaryReasoner",
        "decision_policy": "AcceptAllDecisionPolicy",
        "renderer": "BasicSummaryRenderer",
    }
    for stage, class_name in expected.items():
        assert type(getattr(architecture, stage)).__name__ == class_name
    assert type(architecture.assessors[0]).__name__ == "NoOpAssessor"


@pytest.mark.parametrize(
    "stage",
    [
        "parser",
        "knowledge_builder",
        "retriever",
        "reranker",
        "reasoner",
        "assessors",
        "decision_policy",
        "renderer",
    ],
)
def test_registry_rejects_unknown_components(
    config: ArchitectureConfig, stage: str
) -> None:
    """An unknown identifier names its stage and available alternatives."""
    settings = config.model_dump()
    if stage == "assessors":
        settings[stage][0]["type"] = "unknown"
    else:
        settings[stage]["type"] = "unknown"
    with pytest.raises(ConfigurationError, match="Unknown .* component 'unknown'"):
        ComponentRegistry().build(ArchitectureConfig.model_validate(settings))


@pytest.mark.parametrize(
    "content",
    [
        "",
        "[",
        "[]",
        "name: incomplete",
        "!!python/object:foo {}",
        "name: secret-value\nunknown: secret-value",
    ],
)
def test_invalid_yaml_fails_without_echoing_values(
    tmp_path: Path, content: str
) -> None:
    """YAML syntax, unsafe tags, empty documents, and shape errors are clear."""
    path = tmp_path / "invalid.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ConfigurationError) as caught:
        load_architecture_config(path)
    assert "secret-value" not in str(caught.value)


def test_missing_config_is_actionable(tmp_path: Path) -> None:
    """A missing architecture file is a configuration error with preserved cause."""
    with pytest.raises(ConfigurationError) as caught:
        load_architecture_config(tmp_path / "missing.yaml")
    assert isinstance(caught.value.__cause__, FileNotFoundError)


@pytest.mark.parametrize(
    "override",
    [
        {"provider": "unknown"},
        {"retries": -1},
        {"timeout_seconds": 0},
        {"api_key": "secret"},
        {"base_url": "https://user:secret@example.com/v1"},
        {"base_url": "https://example.com/v1?key=secret"},
    ],
)
def test_model_settings_reject_invalid_or_secret_configuration(
    config: ArchitectureConfig, override: dict[str, object]
) -> None:
    """Only supported credential-free model settings reach runtime construction."""
    settings = config.model_dump()
    settings["reasoner"]["model"].update(override)
    with pytest.raises(ValueError):
        ArchitectureConfig.model_validate(settings)
