"""Shared pipeline management use cases for API, CLI, and runtime clients."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from llm_grounding.application.pipeline_runtime import (
    PipelineRuntimeResolver,
    ResolvedPipeline,
)
from llm_grounding.domain.corpus import Pipeline
from llm_grounding.infrastructure.config.loader import (
    ConfigurationError,
    load_configuration_registry,
)
from llm_grounding.infrastructure.config.models import (
    ModelCatalog,
    PipelineConfig,
    PipelineEmbeddingConfig,
)
from llm_grounding.infrastructure.database.repositories import PostgresRepository
from llm_grounding.ingestion.models import IngestionConfig
from llm_grounding.orchestration.registry import ComponentRegistry


@dataclass(frozen=True)
class PipelineChange:
    """Describe one changed configuration value and its index impact."""

    path: str
    before: object
    after: object
    representation_affecting: bool

    def as_dict(self) -> dict[str, object]:
        """Serialize the change for API and CLI consumers."""
        return {
            "path": self.path,
            "before": self.before,
            "after": self.after,
            "representation_affecting": self.representation_affecting,
        }


@dataclass(frozen=True)
class PipelineUpdateImpact:
    """Summarize dependencies and compatibility before a pipeline update."""

    pipeline_id: str
    current_version: int
    current_config_hash: str
    conversation_count: int
    representation_changed: bool
    compatible: bool
    compatibility_reason: str | None
    changes: tuple[PipelineChange, ...]
    project_count: int = 0

    def as_dict(self) -> dict[str, object]:
        """Serialize authoritative update impact metadata."""
        return {
            "pipeline_id": self.pipeline_id,
            "current_version": self.current_version,
            "current_config_hash": self.current_config_hash,
            "conversation_count": self.conversation_count,
            "project_count": self.project_count,
            "representation_changed": self.representation_changed,
            "compatible": self.compatible,
            "can_migrate": self.compatible,
            "compatibility_reason": self.compatibility_reason,
            "changes": [change.as_dict() for change in self.changes],
        }


class PipelineDependencyError(ValueError):
    """Require an explicit migration or copy when chats use a pipeline."""

    def __init__(self, impact: PipelineUpdateImpact) -> None:
        """Retain the authoritative impact returned by the preflight check."""
        self.impact = impact
        super().__init__(
            f"Pipeline is used by {impact.conversation_count} conversations and "
            f"{impact.project_count} projects; "
            "choose migration or create a copy"
        )


class PipelineCompatibilityError(ValueError):
    """Reject migration that would pair conversations with an incompatible index."""

    def __init__(self, impact: PipelineUpdateImpact) -> None:
        """Retain the compatibility explanation for a structured API error."""
        self.impact = impact
        super().__init__(
            impact.compatibility_reason or "Pipeline representations are incompatible"
        )


class PipelineConcurrencyError(ValueError):
    """Report that an editor attempted to update a stale pipeline snapshot."""


def _with_compatibility_reason(
    impact: PipelineUpdateImpact, reason: str
) -> PipelineCompatibilityError:
    """Create a compatibility error while preserving the complete impact payload."""
    return PipelineCompatibilityError(
        PipelineUpdateImpact(
            pipeline_id=impact.pipeline_id,
            current_version=impact.current_version,
            current_config_hash=impact.current_config_hash,
            conversation_count=impact.conversation_count,
            project_count=impact.project_count,
            representation_changed=impact.representation_changed,
            compatible=False,
            compatibility_reason=reason,
            changes=impact.changes,
        )
    )


def _configuration_changes(
    before: dict[str, object], after: dict[str, object]
) -> tuple[PipelineChange, ...]:
    """Compare JSON-compatible configurations into concise leaf changes."""
    old_values = dict(_flatten_configuration(before))
    new_values = dict(_flatten_configuration(after))
    changes: list[PipelineChange] = []
    for path in sorted(set(old_values) | set(new_values)):
        old = old_values.get(path)
        new = new_values.get(path)
        if old == new:
            continue
        changes.append(
            PipelineChange(
                path=path,
                before=old,
                after=new,
                representation_affecting=path.startswith("representation."),
            )
        )
    return tuple(changes)


def _flatten_configuration(
    value: object, prefix: str = ""
) -> list[tuple[str, object]]:
    """Flatten nested configuration values while retaining list positions."""
    if isinstance(value, dict):
        items: list[tuple[str, object]] = []
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            items.extend(_flatten_configuration(child, child_prefix))
        return items
    if isinstance(value, list):
        return [(prefix, json.dumps(value, sort_keys=True))]
    return [(prefix, value)]


class PipelineService:
    """Manage validated pipeline records and resolve them on demand."""

    def __init__(
        self,
        repository: PostgresRepository,
        catalog: ModelCatalog,
        *,
        registry: ComponentRegistry | None = None,
        runtime: PipelineRuntimeResolver | None = None,
    ) -> None:
        """Inject canonical persistence, model catalog, and optional runtime cache."""
        self._repository = repository
        self._catalog = catalog
        self._registry = registry or ComponentRegistry()
        self._runtime = runtime

    async def list(
        self, *, active_only: bool = False, kind: str | None = None
    ) -> tuple[Pipeline, ...]:
        """List persisted pipeline records, optionally restricted by kind."""
        if kind is not None and kind not in {"full", "ingestion", "query"}:
            raise ValueError(f"Unsupported pipeline kind: {kind}")
        try:
            records = await self._repository.list_pipelines(
                active_only=active_only, kind=kind
            )
        except TypeError:
            records = await self._repository.list_pipelines(active_only=active_only)
            if kind is not None:
                records = tuple(
                    record
                    for record in records
                    if record.kind in {kind, "full"}
                )
        return records

    async def get(self, pipeline_id: str) -> Pipeline:
        """Retrieve one persisted pipeline by stable ID."""
        return await self._repository.get_pipeline(pipeline_id)

    async def default(self, kind: str | None = None) -> Pipeline:
        """Return the active default for a requested pipeline kind."""
        records = await self.list(active_only=True, kind=kind)
        selected = next(
            (record for record in records if record.is_default),
            next(
                (
                    record
                    for record in records
                    if record.name
                    in {"basic_rag_v1", "ingestion_default_v1", "query_basic_v1"}
                ),
                None,
            ),
        )
        if selected is None:
            raise KeyError("No active default pipeline is configured")
        return selected

    async def select(
        self, pipeline_id: str | None = None, *, kind: str | None = None
    ) -> Pipeline:
        """Resolve an explicit active pipeline or the configured default."""
        selected = (
            await self.default(kind)
            if pipeline_id is None
            else await self.get(pipeline_id)
        )
        if not selected.active:
            raise ValueError(f"Pipeline {selected.id!r} is inactive")
        if kind is not None and selected.kind not in {kind, "full"}:
            raise ValueError(
                f"Pipeline {selected.id!r} is a {selected.kind} pipeline, not {kind}"
            )
        return selected

    async def select_ingestion(self, pipeline_id: str | None = None) -> Pipeline:
        """Resolve an active project-level ingestion pipeline."""
        if pipeline_id is not None:
            return await self.select(pipeline_id, kind="ingestion")
        try:
            return await self.select(None, kind="ingestion")
        except KeyError:
            return await self.select(None, kind="full")

    async def select_query(self, pipeline_id: str | None = None) -> Pipeline:
        """Resolve an active chat-level query pipeline."""
        if pipeline_id is not None:
            return await self.select(pipeline_id, kind="query")
        try:
            return await self.select(None, kind="query")
        except KeyError:
            return await self.select(None, kind="full")

    async def create(self, config: PipelineConfig) -> Pipeline:
        """Validate and persist an API-owned pipeline configuration."""
        self.validate(config)
        existing = await self._repository.get_pipeline_by_name(config.name)
        if existing is not None:
            raise ValueError(f"Pipeline name already exists: {config.name}")
        return await self._repository.create_pipeline(
            config,
            source_kind="api",
            representation_hash=config.representation_hash(self._catalog),
        )

    async def impact(
        self, pipeline_id: str, config: PipelineConfig
    ) -> PipelineUpdateImpact:
        """Preflight dependencies, changed fields, and representation compatibility."""
        self.validate(config)
        existing = await self.get(pipeline_id)
        await self._ensure_name_available(pipeline_id, config.name)
        return await self._impact_for(existing, config)

    async def update(
        self,
        pipeline_id: str,
        config: PipelineConfig,
        *,
        expected_config_hash: str | None = None,
        migrate_conversations: bool = False,
    ) -> Pipeline:
        """Update a pipeline safely, optionally migrating dependent conversations."""
        self.validate(config)
        existing = await self.get(pipeline_id)
        if (
            expected_config_hash is not None
            and existing.config_hash != expected_config_hash
        ):
            raise PipelineConcurrencyError(
                "Pipeline changed while the configuration was being edited"
            )
        await self._ensure_name_available(pipeline_id, config.name)
        if config.config_hash == existing.config_hash:
            return existing
        if config.version <= existing.version:
            config = config.model_copy(update={"version": existing.version + 1})
        impact = await self._impact_for(existing, config)
        if impact.conversation_count and not migrate_conversations:
            raise PipelineDependencyError(impact)
        if (
            impact.project_count
            and impact.representation_changed
            and not migrate_conversations
        ):
            raise PipelineDependencyError(impact)
        if migrate_conversations and not impact.compatible:
            raise PipelineCompatibilityError(impact)
        representation_hash = config.representation_hash(self._catalog)
        if migrate_conversations:
            migrate = getattr(self._repository, "update_pipeline_and_migrate", None)
            if migrate is None:
                raise PipelineCompatibilityError(
                    _with_compatibility_reason(
                        impact, "Conversation migration is unavailable."
                    )
                )
            record = await migrate(
                pipeline_id,
                config,
                representation_hash=representation_hash,
                expected_config_hash=expected_config_hash or existing.config_hash,
            )
        else:
            kwargs: dict[str, object] = {"representation_hash": representation_hash}
            if expected_config_hash is not None:
                kwargs["expected_config_hash"] = expected_config_hash
            try:
                record = await self._repository.update_pipeline(
                    pipeline_id, config, **kwargs
                )
            except ValueError as exc:
                if "changed while" in str(exc):
                    raise PipelineConcurrencyError(str(exc)) from exc
                raise
        if self._runtime is not None:
            self._runtime.invalidate(pipeline_id)
        return record

    async def _impact_for(
        self, existing: Pipeline, config: PipelineConfig
    ) -> PipelineUpdateImpact:
        """Build impact from an already-loaded pipeline without another lookup."""
        count_method = getattr(
            self._repository, "count_pipeline_conversations", None
        )
        conversation_count = (
            int(await count_method(existing.id)) if count_method is not None else 0
        )
        project_count_method = getattr(
            self._repository, "count_pipeline_projects", None
        )
        project_count = (
            int(await project_count_method(existing.id))
            if project_count_method is not None
            else 0
        )
        representation_changed = existing.kind != config.kind or (
            existing.kind in {"full", "ingestion"}
            and config.representation_hash(self._catalog)
            != existing.representation_hash
        )
        return PipelineUpdateImpact(
            pipeline_id=existing.id,
            current_version=existing.version,
            current_config_hash=existing.config_hash,
            conversation_count=conversation_count,
            representation_changed=representation_changed,
            compatible=not (
                (conversation_count > 0 or project_count > 0)
                and representation_changed
            ),
            compatibility_reason=(
                "Existing conversations or project indexes use the current "
                "representation; re-indexing is required before migration."
                if (conversation_count > 0 or project_count > 0)
                and representation_changed
                else None
            ),
            changes=_configuration_changes(
                existing.config, config.model_dump(mode="json")
            ),
            project_count=project_count,
        )

    async def _ensure_name_available(self, pipeline_id: str, name: str) -> None:
        """Reject updates that would collide with another stable pipeline name."""
        existing = await self._repository.get_pipeline_by_name(name)
        if existing is not None and existing.id != pipeline_id:
            raise ValueError(f"Pipeline name already exists: {name}")

    async def deactivate(self, pipeline_id: str) -> Pipeline:
        """Deactivate a pipeline while retaining its persisted configuration."""
        record = await self._repository.deactivate_pipeline(pipeline_id)
        if self._runtime is not None:
            self._runtime.invalidate(pipeline_id)
        return record

    def validate(self, config: PipelineConfig) -> PipelineConfig:
        """Validate stage identifiers, model references, and compatibility."""
        self._registry.validate_pipeline(config, self._catalog)
        return config

    async def resolve(self, pipeline_id: str) -> ResolvedPipeline:
        """Resolve a stored pipeline through the shared bounded runtime cache."""
        if self._runtime is None:
            raise RuntimeError("Pipeline runtime resolution is not configured")
        return await self._runtime.resolve(pipeline_id)

    async def resolve_ingestion(self, pipeline_id: str) -> IngestionConfig:
        """Resolve a persisted ingestion definition into effective settings."""
        record = await self.select_ingestion(pipeline_id)
        config = PipelineConfig.model_validate(record.config)
        self.validate(config)
        if record.representation_hash != config.representation_hash(self._catalog):
            raise ValueError(
                f"Pipeline {record.id!r} has a stale ingestion representation hash"
            )
        return config.resolve_ingestion(self._catalog)

    @property
    def runtime(self) -> PipelineRuntimeResolver | None:
        """Return the shared resolver used by API, CLI, and chat execution."""
        return self._runtime

    @property
    def registry(self) -> ComponentRegistry:
        """Expose the closed registry to shared compatibility validators."""
        return self._registry

    def summaries(self, records: tuple[Pipeline, ...]) -> list[dict[str, object]]:
        """Serialize list metadata without including complete JSON configuration."""
        return [
            {
                "id": item.id,
                "name": item.name,
                "version": item.version,
                "schema_version": item.schema_version,
                "description": item.description,
                "config_hash": item.config_hash,
                "representation_hash": item.representation_hash,
                "kind": item.kind,
                "source_kind": item.source_kind,
                "active": item.active,
                "is_default": item.is_default,
                "created_at": item.created_at,
                "updated_at": item.updated_at,
            }
            for item in records
        ]

    def catalog(self) -> dict[str, object]:
        """Return registered stages, schemas, and compatibility constraints."""
        try:
            standalone = load_configuration_registry()
            components: dict[str, list[str]] = {}
            for category, component in standalone.components.values():
                components.setdefault(category, []).append(component.id)
            for category in components:
                components[category].sort()
            configuration_ids: dict[str, object] = {
                "models": sorted(standalone.models),
                "components": components,
            }
        except ConfigurationError:
            configuration_ids = {"models": [], "components": {}}
        return {
            "schema_version": 1,
            "configuration_ids": configuration_ids,
            "pipeline_kinds": {
                "ingestion": [
                    "parser",
                    "knowledge_builder",
                    "chunker",
                    "embedding",
                    "vector_backend",
                ],
                "query": [
                    "retriever",
                    "reranker",
                    "reasoner",
                    "assessors",
                    "decision_policy",
                    "renderer",
                ],
            },
            "stages": {
                "parser": self._stage(self._registry.parsers.keys(), "ComponentConfig"),
                "knowledge_builder": self._stage(
                    self._registry.knowledge_builders.keys(), "ComponentConfig"
                ),
                "chunker": self._stage(self._registry.chunkers.keys(), "ChunkerConfig"),
                "embedding": {
                    "implementations": sorted(self._catalog.embedding_models),
                    "schema": PipelineEmbeddingConfig.model_json_schema(),
                },
                "vector_backend": self._stage(
                    self._registry.vector_backends, "VectorBackendConfig"
                ),
                "retriever": self._stage(
                    self._registry.retrievers.keys(), "RetrieverConfig"
                ),
                "reranker": self._stage(
                    self._registry.rerankers.keys(), "RerankerConfig"
                ),
                "reasoner": self._stage(
                    self._registry.pipeline_reasoners, "PipelineReasonerConfig"
                ),
                "assessor": self._stage(
                    self._registry.pipeline_assessors, "ComponentConfig"
                ),
                "decision_policy": self._stage(
                    self._registry.pipeline_decision_policies, "ComponentConfig"
                ),
                "renderer": self._stage(
                    self._registry.pipeline_renderers, "ComponentConfig"
                ),
            },
            "compatibility": {
                "vector_retriever_requires": {"vector_backend": ["qdrant"]},
                "retriever_capabilities": {
                    name: sorted(self._registry.query_capabilities(name))
                    for name in self._registry.retrievers
                },
                "basic_grounded_requires": {
                    "assessor": ["source_reference"],
                    "decision_policy": ["basic_grounding"],
                    "renderer": ["grounded_answer"],
                },
            },
        }

    def models(self) -> dict[str, object]:
        """Return catalog model metadata without credentials or secret fields."""
        return {
            "schema_version": self._catalog.schema_version,
            "embedding": {
                name: model.model_dump(mode="json")
                for name, model in self._catalog.embedding_models.items()
            },
            "reasoning": {
                name: model.model_dump(mode="json")
                for name, model in self._catalog.reasoning_models.items()
            },
        }

    @staticmethod
    def _stage(implementations: Any, schema_name: str) -> dict[str, object]:
        """Build one catalog entry from registered names and a typed schema."""
        from llm_grounding.infrastructure.config import models

        schema = getattr(models, schema_name).model_json_schema()
        return {
            "implementations": sorted(implementations),
            "schema": schema,
        }
