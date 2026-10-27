"""Verify standardized component schemas and controlled runtime overrides."""

from pathlib import Path
from shutil import copytree

import pytest

from llm_grounding.application.component_sync import ComponentSynchronizer
from llm_grounding.infrastructure.config.loader import (
    ConfigurationError,
    load_configuration_registry,
    load_pipeline_config,
)
from llm_grounding.infrastructure.config.overrides import (
    OverrideValidationError,
    apply_query_overrides,
)


def test_component_registry_validates_instances_and_excludes_schemas() -> None:
    """All shipped instances load while schema files remain non-components."""
    registry = load_configuration_registry()

    assert len(registry.components) == 22
    assert "schema" not in registry.components
    retriever = registry.component(
        "retriever", "retriever_sparse_postgres_fts"
    )
    assert retriever.type == "sparse"
    assert retriever.implementation == "postgres_fts"
    assert retriever.config["top_k"] == 20


def test_invalid_component_yaml_fails_family_schema_validation(tmp_path: Path) -> None:
    """Missing standard fields and unknown implementations fail before resolution."""
    components = tmp_path / "components"
    copytree(Path("config/components"), components)
    invalid = components / "parser" / "invalid.yaml"
    invalid.write_text(
        "id: parser_invalid\ntype: parser\nimplementation: unknown\nconfig: {}\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="Invalid component configuration"):
        load_configuration_registry(components_directory=components)


def test_override_precedence_and_scope_metadata_are_enforced() -> None:
    """Pipeline, chat, and request layers merge without mutating the base config."""
    config = load_pipeline_config(Path("config/pipelines/query_basic_v1.yaml"))

    effective, provenance = apply_query_overrides(
        config,
        (
            ("pipeline", {"retriever": {"top_k": 20}}),
            ("chat", {"retriever": {"top_k": 10}}),
            ("request", {"retriever": {"top_k": 3}}),
        ),
    )

    assert config.query is not None
    assert config.query.retriever.top_k == 8
    assert effective.query is not None
    assert effective.query.retriever.top_k == 3
    assert provenance["chat"] == {"retriever": {"top_k": 10}}

    with pytest.raises(OverrideValidationError, match="cannot be overridden"):
        apply_query_overrides(
            config,
            (("request", {"retriever": {"query_embedder": "embedding_001"}}),),
        )

    with pytest.raises(OverrideValidationError, match="Unknown retriever"):
        apply_query_overrides(
            config,
            (("chat", {"retriever": {"unknown": 1}}),),
        )


async def test_component_synchronizer_persists_validated_definitions() -> None:
    """Startup synchronization sends schema-validated definitions to PostgreSQL."""

    class Repository:
        def __init__(self) -> None:
            self.rows: list[tuple[str, str]] = []

        async def upsert_component_definition(
            self, category, definition, *, source_path, source_fingerprint
        ) -> None:
            self.rows.append((category, definition.id))

    repository = Repository()
    result = await ComponentSynchronizer(repository).synchronize()  # type: ignore[arg-type]

    assert result.imported == 22
    assert len(repository.rows) == 22
    assert ("retriever", "retriever_sparse_postgres_fts") in repository.rows
