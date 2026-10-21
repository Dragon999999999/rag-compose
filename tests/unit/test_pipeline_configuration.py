"""Verify unified pipeline validation and representation compatibility identities."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from llm_grounding.application.pipeline_sync import PipelineSynchronizer
from llm_grounding.domain.corpus import Pipeline
from llm_grounding.infrastructure.config.loader import (
    ConfigurationError,
    load_model_catalog,
    load_pipeline_config,
)
from llm_grounding.infrastructure.database.models import PipelineRow
from llm_grounding.orchestration.registry import ComponentRegistry


def test_all_checked_in_pipelines_validate_against_real_catalog() -> None:
    """Every shipped pipeline resolves only registered components and models."""
    catalog = load_model_catalog(Path("configs/models.yaml"))
    paths = sorted(Path("configs/pipelines").glob("*.yaml"))

    assert len(paths) == 4
    configs = [load_pipeline_config(path, catalog=catalog) for path in paths]
    assert {config.name for config in configs} == {
        "basic_rag_v1",
        "basic_rag_small_chunks",
        "basic_rag_large_chunks",
        "basic_rag_fast_reasoning",
    }


def test_representation_changes_produce_distinct_index_identity() -> None:
    """Chunking changes cannot silently reuse an existing representation index."""
    catalog = load_model_catalog(Path("configs/models.yaml"))
    base = load_pipeline_config(
        Path("configs/pipelines/basic_rag_v1.yaml"), catalog=catalog
    )
    changed = base.model_copy(
        update={
            "representation": base.representation.model_copy(
                update={
                    "chunker": base.representation.chunker.model_copy(
                        update={"target_chars": 600}
                    )
                }
            )
        }
    )

    assert base.representation_hash(catalog) != changed.representation_hash(catalog)


def test_pipeline_loader_rejects_unknown_models_without_echoing_secret(
    tmp_path: Path,
) -> None:
    """Catalog references remain closed and invalid values are not leaked."""
    source = Path("configs/pipelines/basic_rag_v1.yaml").read_text(encoding="utf-8")
    path = tmp_path / "invalid.yaml"
    path.write_text(
        source.replace("model_ref: embedding_001", "model_ref: secret-model"),
        encoding="utf-8",
    )

    with pytest.raises(
        ConfigurationError, match="Unknown embedding model reference"
    ) as caught:
        load_pipeline_config(path)

    assert "secret-model" not in str(caught.value)


def test_pipeline_row_persists_jsonb_configuration_and_ownership_fields() -> None:
    """The ORM contract includes source ownership and representation identity."""
    columns = PipelineRow.__table__.c

    assert columns.config_json.type.__class__.__name__ == "JSONB"
    assert {"config_hash", "representation_hash", "source_kind"}.issubset(
        columns.keys()
    )
    assert {"active", "is_default", "updated_at"}.issubset(columns.keys())


def test_registry_keeps_pipeline_registration_closed() -> None:
    """A pipeline cannot select arbitrary Python import paths."""
    config = load_pipeline_config(Path("configs/pipelines/basic_rag_v1.yaml"))
    settings = config.model_dump()
    settings["query"]["renderer"]["type"] = "python:os.system"

    with pytest.raises(ConfigurationError, match="Unknown renderer component"):
        ComponentRegistry().validate_pipeline(
            type(config).model_validate(settings), load_model_catalog()
        )


class RecordingPipelineRepository:
    """Record synchronizer calls without requiring a PostgreSQL service."""

    def __init__(self) -> None:
        """Start with an empty persisted pipeline collection."""
        self.records: dict[str, Pipeline] = {}

    async def get_pipeline_by_name(self, name: str) -> Pipeline | None:
        """Return the current record for a stable pipeline name."""
        return self.records.get(name)

    async def upsert_pipeline_from_yaml(
        self,
        config,
        *,
        source_path: str,
        source_fingerprint: str,
        representation_hash: str,
    ) -> Pipeline:
        """Apply the synchronizer's YAML-owned upsert contract."""
        previous = self.records.get(config.name)
        if previous is not None and previous.source_kind == "api":
            return previous
        now = datetime.now(UTC)
        record = Pipeline(
            previous.id if previous is not None else config.name,
            config.name,
            config.version,
            config.schema_version,
            config.description,
            config.model_dump(mode="json"),
            config.config_hash,
            representation_hash,
            "yaml",
            source_path,
            source_fingerprint,
            True,
            config.is_default,
            previous.created_at if previous is not None else now,
            now,
        )
        self.records[config.name] = record
        return record


async def test_startup_synchronizer_discovers_and_fingerprints_yaml(
    tmp_path: Path,
) -> None:
    """Startup imports new files and remains idempotent on the next pass."""
    source = Path("configs/pipelines/basic_rag_v1.yaml")
    (tmp_path / source.name).write_text(
        source.read_text(encoding="utf-8"), encoding="utf-8"
    )
    repository = RecordingPipelineRepository()
    synchronizer = PipelineSynchronizer(repository, tmp_path, load_model_catalog())

    first = await synchronizer.synchronize()
    second = await synchronizer.synchronize()

    assert first.imported == 1
    assert first.updated == 0
    assert first.pipelines[0].source_fingerprint
    assert second.imported == 0
    assert second.updated == 0
