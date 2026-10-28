"""Test pipeline management services and runtime composition resolution."""

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from llm_grounding.api.main import create_api
from llm_grounding.application.pipeline_runtime import (
    PipelineResolutionError,
    PipelineRuntimeResolver,
)
from llm_grounding.application.pipeline_service import (
    PipelineDependencyError,
    PipelineService,
)
from llm_grounding.domain.corpus import Pipeline
from llm_grounding.infrastructure.config.loader import (
    load_ingestion_config,
    load_model_catalog,
    load_pipeline_config,
    pipeline_source_fingerprint,
)
from llm_grounding.infrastructure.config.models import ModelCatalog
from llm_grounding.infrastructure.vector.memory import InMemoryVectorStore


class PipelineRepository:
    """Provide async pipeline persistence for service and resolver tests."""

    def __init__(self, catalog: ModelCatalog) -> None:
        """Initialize an empty repository with the supplied model catalog."""
        self.catalog = catalog
        self.records: dict[str, Pipeline] = {}
        self.conversation_count = 0
        self.migrated_snapshots: list[tuple[int, str, str]] = []

    async def list_pipelines(
        self, *, active_only: bool = False
    ) -> tuple[Pipeline, ...]:
        """List records in insertion order, optionally excluding inactive rows."""
        records = tuple(self.records.values())
        return tuple(record for record in records if record.active or not active_only)

    async def get_pipeline(self, pipeline_id: str) -> Pipeline:
        """Return one record or the repository's normal missing-record error."""
        try:
            return self.records[pipeline_id]
        except KeyError as exc:
            raise KeyError(f"Unknown pipeline: {pipeline_id}") from exc

    async def get_pipeline_by_name(self, name: str) -> Pipeline | None:
        """Find a record by its stable configured name."""
        return next(
            (record for record in self.records.values() if record.name == name), None
        )

    async def create_pipeline(
        self,
        config,
        *,
        source_kind: str,
        representation_hash: str,
    ) -> Pipeline:
        """Persist a newly created pipeline record."""
        identifier = f"pipeline-{len(self.records) + 1}"
        now = datetime.now(UTC)
        record = Pipeline(
            identifier,
            config.name,
            config.version,
            config.schema_version,
            config.description,
            config.model_dump(mode="json"),
            config.config_hash,
            representation_hash,
            source_kind,
            None,
            None,
            True,
            config.is_default,
            now,
            now,
        )
        self.records[identifier] = record
        return record

    async def update_pipeline(
        self,
        pipeline_id: str,
        config,
        *,
        representation_hash: str,
    ) -> Pipeline:
        """Replace a record's structured configuration and mark it API-owned."""
        current = await self.get_pipeline(pipeline_id)
        updated = replace(
            current,
            name=config.name,
            version=config.version,
            schema_version=config.schema_version,
            description=config.description,
            config=config.model_dump(mode="json"),
            config_hash=config.config_hash,
            representation_hash=representation_hash,
            source_kind="api",
            source_path=None,
            source_fingerprint=None,
            is_default=config.is_default,
            updated_at=datetime.now(UTC),
        )
        self.records[pipeline_id] = updated
        return updated

    async def count_pipeline_conversations(self, pipeline_id: str) -> int:
        """Return the number of conversations using the selected pipeline."""
        await self.get_pipeline(pipeline_id)
        return self.conversation_count

    async def update_pipeline_and_migrate(
        self,
        pipeline_id: str,
        config,
        *,
        representation_hash: str,
        expected_config_hash: str,
    ) -> Pipeline:
        """Apply a compatible update and record migrated chat snapshots."""
        current = await self.get_pipeline(pipeline_id)
        if current.config_hash != expected_config_hash:
            raise ValueError(
                "Pipeline changed while the configuration was being edited"
            )
        if current.representation_hash != representation_hash:
            raise ValueError("The updated representation is incompatible")
        updated = await self.update_pipeline(
            pipeline_id, config, representation_hash=representation_hash
        )
        self.migrated_snapshots = [
            (updated.version, updated.config_hash, updated.representation_hash)
        ] * self.conversation_count
        return updated

    async def deactivate_pipeline(self, pipeline_id: str) -> Pipeline:
        """Retain a record while making it unavailable for active selection."""
        current = await self.get_pipeline(pipeline_id)
        updated = replace(current, active=False, updated_at=datetime.now(UTC))
        self.records[pipeline_id] = updated
        return updated


class RecordingRuntime:
    """Record service-triggered runtime invalidations."""

    def __init__(self) -> None:
        """Start with no invalidation requests."""
        self.invalidated: list[str] = []

    def invalidate(self, pipeline_id: str) -> None:
        """Record the pipeline whose cached runtime must be discarded."""
        self.invalidated.append(pipeline_id)


@pytest.mark.asyncio
async def test_repairs_only_verified_legacy_ingestion_hashes() -> None:
    """Hash repair previews and applies only metadata for an unchanged YAML source."""
    catalog = load_model_catalog()
    path = Path("config/pipelines/ingestion_default_v1.yaml").resolve()
    config = load_pipeline_config(path, catalog=catalog).with_resolved_catalog(catalog)
    now = datetime.now(UTC)
    record = Pipeline(
        "legacy-ingestion",
        config.name,
        config.version,
        config.schema_version,
        config.description,
        config.model_dump(mode="json"),
        config.legacy_identity(),
        config.representation.legacy_identity(catalog),  # type: ignore[union-attr]
        "yaml",
        str(path),
        pipeline_source_fingerprint(path),
        True,
        config.is_default,
        now,
        now,
        "ingestion",
    )

    class RepairRepository:
        def __init__(self) -> None:
            self.repairs: tuple[tuple[str, str, str, str, str], ...] = ()

        async def list_pipelines(self, **_kwargs: object) -> tuple[Pipeline, ...]:
            return (record,)

        async def repair_ingestion_hashes(
            self, repairs: tuple[tuple[str, str, str, str, str], ...]
        ) -> dict[str, int]:
            self.repairs = repairs
            return {
                "pipelines": 1,
                "pipeline_versions": 1,
                "projects": 0,
                "index_capabilities": 0,
            }

    repository = RepairRepository()
    service = PipelineService(repository, catalog)  # type: ignore[arg-type]

    preview = await service.repair_ingestion_hashes(dry_run=True)
    assert len(preview.repairs) == 1
    assert repository.repairs == ()

    applied = await service.repair_ingestion_hashes()
    assert applied.updated["pipelines"] == 1
    assert repository.repairs == (
        (
            record.id,
            record.config_hash,
            config.config_hash,
            record.representation_hash,
            config.representation_hash(catalog),
        ),
    )


async def test_pipeline_service_crud_validates_and_invalidates_runtime_async() -> None:
    """API-owned CRUD uses one validated service and clears stale runtimes."""
    catalog = load_model_catalog()
    config = load_pipeline_config(
        Path("config/pipelines/query_basic_v1.yaml"), catalog=catalog
    )
    repository = PipelineRepository(catalog)
    runtime = RecordingRuntime()
    service = PipelineService(repository, catalog, runtime=runtime)  # type: ignore[arg-type]

    created = await service.create(config)
    changed = config.model_copy(update={"description": "API-owned variant"})
    updated = await service.update(created.id, changed)
    deactivated = await service.deactivate(created.id)

    assert created.source_kind == "api"
    assert updated.description == "API-owned variant"
    assert runtime.invalidated == [created.id, created.id]
    assert deactivated.active is False
    assert await service.list(active_only=True) == ()


async def test_used_pipeline_reports_changes_and_requires_an_explicit_choice() -> None:
    """Dependent conversations prevent an implicit in-place update."""
    catalog = load_model_catalog()
    config = load_pipeline_config(
        Path("config/pipelines/query_basic_v1.yaml"), catalog=catalog
    )
    repository = PipelineRepository(catalog)
    service = PipelineService(repository, catalog)  # type: ignore[arg-type]
    created = await service.create(config)
    repository.conversation_count = 2
    changed = config.model_copy(
        update={
            "name": config.name,
            "query": config.query.model_copy(
                update={
                    "retriever": config.query.retriever.model_copy(update={"top_k": 4})
                }
            ),
        }
    )

    impact = await service.impact(created.id, changed)

    assert impact.conversation_count == 2
    assert impact.compatible is True
    assert any(change.path == "query.retriever.top_k" for change in impact.changes)
    with pytest.raises(PipelineDependencyError):
        await service.update(created.id, changed)


async def test_used_pipeline_can_migrate_query_changes_but_not_representation_changes(
) -> None:
    """Migrate query changes while blocking representation changes."""
    catalog = load_model_catalog()
    config = load_pipeline_config(
        Path("config/pipelines/query_basic_v1.yaml"), catalog=catalog
    )
    repository = PipelineRepository(catalog)
    service = PipelineService(repository, catalog)  # type: ignore[arg-type]
    created = await service.create(config)
    repository.conversation_count = 2

    query_change = config.model_copy(
        update={
            "query": config.query.model_copy(
                update={
                    "retriever": config.query.retriever.model_copy(update={"top_k": 4})
                }
            )
        }
    )
    migrated = await service.update(
        created.id,
        query_change,
        expected_config_hash=created.config_hash,
        migrate_conversations=True,
    )
    assert migrated.version == created.version + 1
    assert len(repository.migrated_snapshots) == 2


async def test_runtime_resolver_builds_distinct_cached_pipeline_configurations() -> (
    None
):
    """Different stored query parameters create distinct runnable compositions."""
    catalog = load_model_catalog()
    ingestion = load_ingestion_config()
    base = load_pipeline_config(
        Path("config/pipelines/query_basic_v1.yaml"), catalog=catalog
    )
    repository = PipelineRepository(catalog)
    for index, top_k in enumerate((3, 7, 11), start=1):
        config = base.model_copy(
            update={
                "name": f"runtime_variant_{index}",
                "is_default": index == 1,
                "query": base.query.model_copy(
                    update={
                        "retriever": base.query.retriever.model_copy(
                            update={"top_k": top_k}
                        )
                    }
                ),
            }
        )
        now = datetime.now(UTC)
        repository.records[str(index)] = Pipeline(
            str(index),
            config.name,
            config.version,
            config.schema_version,
            config.description,
            config.model_dump(mode="json"),
            config.config_hash,
            config.representation_hash(catalog),
            "api",
            None,
            None,
            True,
            config.is_default,
            now,
            now,
        )

    resolver = PipelineRuntimeResolver(
        repository,
        InMemoryVectorStore(),
        catalog,
        max_cache_size=2,  # type: ignore[arg-type]
    )
    first = await resolver.resolve("1", ingestion=ingestion)
    cached = await resolver.resolve("1", ingestion=ingestion)
    second = await resolver.resolve("2", ingestion=ingestion)
    third = await resolver.resolve("3", ingestion=ingestion)

    assert first is cached
    assert first.architecture.retriever.top_k == 3
    assert second.architecture.retriever.top_k == 7
    assert third.architecture.retriever.top_k == 11
    assert resolver.cache_size == 2


async def test_runtime_resolver_rebuilds_after_persisted_hash_changes() -> None:
    """A changed persisted configuration cannot reuse the old runtime instance."""
    catalog = load_model_catalog()
    ingestion = load_ingestion_config()
    config = load_pipeline_config(
        Path("config/pipelines/query_basic_v1.yaml"), catalog=catalog
    )
    now = datetime.now(UTC)
    record = Pipeline(
        "one",
        config.name,
        config.version,
        config.schema_version,
        config.description,
        config.model_dump(mode="json"),
        config.config_hash,
        config.representation_hash(catalog),
        "api",
        None,
        None,
        True,
        True,
        now,
        now,
    )
    repository = PipelineRepository(catalog)
    repository.records[record.id] = record
    resolver = PipelineRuntimeResolver(
        repository,
        InMemoryVectorStore(),
        catalog,  # type: ignore[arg-type]
    )
    original = await resolver.resolve(record.id, ingestion=ingestion)

    changed = config.model_copy(
        update={
            "query": config.query.model_copy(
                update={
                    "retriever": config.query.retriever.model_copy(update={"top_k": 4})
                }
            )
        }
    )
    repository.records[record.id] = replace(
        record,
        config=changed.model_dump(mode="json"),
        config_hash=changed.config_hash,
        representation_hash=changed.representation_hash(catalog),
    )

    refreshed = await resolver.resolve(record.id, ingestion=ingestion)

    assert refreshed is not original
    assert refreshed.architecture.retriever.top_k == 4


async def test_runtime_resolver_rejects_incompatible_representation_identity() -> None:
    """A stale index identity cannot be silently paired with a pipeline runtime."""
    catalog = load_model_catalog()
    config = load_pipeline_config(
        Path("config/pipelines/query_basic_v1.yaml"), catalog=catalog
    )
    now = datetime.now(UTC)
    record = Pipeline(
        "stale",
        config.name,
        config.version,
        config.schema_version,
        config.description,
        config.model_dump(mode="json"),
        config.config_hash,
        "stale-representation-hash",
        "api",
        None,
        None,
        True,
        True,
        now,
        now,
    )
    repository = PipelineRepository(catalog)
    repository.records[record.id] = record
    resolver = PipelineRuntimeResolver(
        repository,
        InMemoryVectorStore(),
        catalog,  # type: ignore[arg-type]
    )

    with pytest.raises(PipelineResolutionError):
        await resolver.resolve(record.id)


async def test_pipeline_management_api_exposes_crud_catalog_and_models() -> None:
    """HTTP routes delegate CRUD and discovery to the shared pipeline service."""
    catalog = load_model_catalog()
    repository = PipelineRepository(catalog)
    service = PipelineService(repository, catalog)  # type: ignore[arg-type]
    application = SimpleNamespace(
        pipeline_service=service,
        evaluation=SimpleNamespace(corpus_service=object(), embedding_store=object()),
        repository=object(),
        _vectors=object(),
    )
    api = create_api(application)  # type: ignore[arg-type]
    config = load_pipeline_config(
        Path("config/pipelines/query_basic_v1.yaml"), catalog=catalog
    )

    transport = httpx.ASGITransport(app=api)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        created_response = await client.post(
            "/api/pipelines", json=config.model_dump(mode="json")
        )
        pipeline_id = created_response.json()["id"]
        changed_config = config.model_copy(update={"description": "updated"})
        updated_response = await client.put(
            f"/api/pipelines/{pipeline_id}",
            json=changed_config.model_dump(mode="json"),
        )
        listed_response = await client.get("/api/pipelines")
        detail_response = await client.get(f"/api/pipelines/{pipeline_id}")
        catalog_response = await client.get("/api/catalog")
        models_response = await client.get("/api/models")
        deleted_response = await client.delete(f"/api/pipelines/{pipeline_id}")

    assert created_response.status_code == 201
    assert updated_response.status_code == 200
    assert listed_response.status_code == 200
    assert detail_response.json()["config"] == changed_config.model_copy(
        update={"version": 2}
    ).model_dump(mode="json")
    assert "vector_backend" in catalog_response.json()["stages"]
    assert "embedding_001" in models_response.json()["embedding"]
    assert deleted_response.status_code == 204
