"""Verify unified pipeline validation and representation compatibility identities."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from llm_grounding.application.pipeline_compatibility import (
    IndexCapabilities,
    validate_query_compatibility,
)
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
    catalog = load_model_catalog()
    paths = sorted(Path("config/pipelines").glob("*.yaml"))

    assert len(paths) == 2
    configs = [load_pipeline_config(path, catalog=catalog) for path in paths]
    assert {config.name for config in configs} == {
        "ingestion_default_v1",
        "query_basic_v1",
    }


def test_representation_changes_produce_distinct_index_identity() -> None:
    """Chunking changes cannot silently reuse an existing representation index."""
    catalog = load_model_catalog()
    base = load_pipeline_config(
        Path("config/pipelines/ingestion_default_v1.yaml"), catalog=catalog
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



def test_ingestion_and_query_pipeline_specs_resolve_separately() -> None:
    """Specialized definitions compose through a project's effective ingestion."""
    catalog = load_model_catalog()
    ingestion = load_pipeline_config(
        Path("config/pipelines/ingestion_default_v1.yaml"), catalog=catalog
    )
    query = load_pipeline_config(
        Path("config/pipelines/query_basic_v1.yaml"), catalog=catalog
    )

    assert ingestion.kind == "ingestion"
    assert ingestion.query is None
    assert query.kind == "query"
    assert query.representation is None
    effective_ingestion = ingestion.resolve_ingestion(catalog)
    architecture, resolved_ingestion = query.resolve_query(catalog, effective_ingestion)
    assert architecture.retriever.type == "vector"
    assert resolved_ingestion.identity() == effective_ingestion.identity()


def test_query_compatibility_uses_index_capabilities_not_dimension_alone() -> None:
    """A query pipeline requires the actual vector capability and embedding identity."""
    catalog = load_model_catalog()
    query = load_pipeline_config(
        Path("config/pipelines/query_basic_v1.yaml"), catalog=catalog
    )
    ingestion = load_pipeline_config(
        Path("config/pipelines/ingestion_default_v1.yaml"), catalog=catalog
    ).resolve_ingestion(catalog)
    compatible = validate_query_compatibility(
        query,
        IndexCapabilities.from_ingestion(ingestion),
        ComponentRegistry(),
    )
    dimension_only = IndexCapabilities(frozenset({"chunk_vectors"}), None, 384)
    incompatible = validate_query_compatibility(
        query, dimension_only, ComponentRegistry()
    )

    assert compatible.compatible
    assert not incompatible.compatible
    assert "embedding_space" in incompatible.missing_capabilities


def test_pipeline_loader_rejects_unknown_components_without_echoing_secret(
    tmp_path: Path,
) -> None:
    """Component references remain closed and invalid values are not leaked."""
    source = Path("config/pipelines/ingestion_default_v1.yaml").read_text(
        encoding="utf-8"
    )
    path = tmp_path / "invalid.yaml"
    path.write_text(
        source.replace("parser: parser_pymupdf", "parser: secret-model"),
        encoding="utf-8",
    )

    with pytest.raises(
        ConfigurationError, match="Invalid pipeline component or model reference"
    ) as caught:
        load_pipeline_config(path)

    assert "secret-model" not in str(caught.value)


def test_pipeline_row_persists_jsonb_configuration_and_ownership_fields() -> None:
    """The ORM contract includes source ownership and representation identity."""
    columns = PipelineRow.__table__.c

    assert columns.config_json.type.__class__.__name__ == "JSONB"
    assert {"config_hash", "representation_hash", "source_kind", "kind"}.issubset(
        columns.keys()
    )
    assert {"active", "is_default", "updated_at"}.issubset(columns.keys())


def test_registry_keeps_pipeline_registration_closed() -> None:
    """A pipeline cannot select arbitrary Python import paths."""
    config = load_pipeline_config(Path("config/pipelines/query_basic_v1.yaml"))
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
        self.versions: dict[tuple[str, int], str] = {}

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
        pipeline_id = previous.id if previous is not None else config.name
        version_key = (pipeline_id, config.version)
        stored_hash = self.versions.get(version_key)
        if stored_hash is not None and stored_hash != config.config_hash:
            raise ValueError(
                f"Pipeline {pipeline_id} version {config.version} already contains "
                "a different configuration; increment the pipeline version"
            )
        now = datetime.now(UTC)
        record = Pipeline(
            pipeline_id,
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
        self.versions[version_key] = config.config_hash
        return record


async def test_startup_synchronizer_discovers_and_fingerprints_yaml(
    tmp_path: Path,
) -> None:
    """Startup imports new files and remains idempotent on the next pass."""
    source = Path("config/pipelines/query_basic_v1.yaml")
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


async def test_startup_sync_rejects_changed_yaml_without_version_bump(
    tmp_path: Path,
) -> None:
    """A changed YAML document cannot overwrite an immutable version snapshot."""
    source = Path("config/pipelines/query_basic_v1.yaml")
    pipeline_path = tmp_path / source.name
    pipeline_path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    repository = RecordingPipelineRepository()
    synchronizer = PipelineSynchronizer(repository, tmp_path, load_model_catalog())

    await synchronizer.synchronize()
    pipeline_path.write_text(
        pipeline_path.read_text(encoding="utf-8").replace(
            "description: Default query pipeline for indexed project documents.",
            "description: Changed query pipeline for indexed project documents.",
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"Pipeline query_basic_v1 version 1 already contains a different",
    ):
        await synchronizer.synchronize()

    assert repository.records["query_basic_v1"].version == 1
    assert (
        repository.records["query_basic_v1"].config["query"]["retriever"]["top_k"]
        == 8
    )


async def test_startup_sync_accepts_changed_yaml_after_version_bump(
    tmp_path: Path,
) -> None:
    """A changed YAML document imports normally once its version is incremented."""
    source = Path("config/pipelines/query_basic_v1.yaml")
    pipeline_path = tmp_path / source.name
    pipeline_path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    repository = RecordingPipelineRepository()
    synchronizer = PipelineSynchronizer(repository, tmp_path, load_model_catalog())

    await synchronizer.synchronize()
    changed = pipeline_path.read_text(encoding="utf-8")
    pipeline_path.write_text(
        changed.replace("\nversion: 1\n", "\nversion: 2\n", 1).replace(
            "description: Default query pipeline for indexed project documents.",
            "description: Changed query pipeline for indexed project documents.",
        ),
        encoding="utf-8",
    )

    updated = await synchronizer.synchronize()
    repeated = await synchronizer.synchronize()

    assert updated.updated == 1
    assert repeated.updated == 0
    assert repository.records["query_basic_v1"].version == 2
    assert (
        repository.records["query_basic_v1"].config["query"]["retriever"]["top_k"]
        == 8
    )
