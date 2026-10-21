"""Shared pipeline management use cases for API, CLI, and runtime clients."""

from __future__ import annotations

from typing import Any

from llm_grounding.application.pipeline_runtime import (
    PipelineRuntimeResolver,
    ResolvedPipeline,
)
from llm_grounding.domain.corpus import Pipeline
from llm_grounding.infrastructure.config.models import ModelCatalog, PipelineConfig
from llm_grounding.infrastructure.database.repositories import PostgresRepository
from llm_grounding.orchestration.registry import ComponentRegistry


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

    async def list(self, *, active_only: bool = False) -> tuple[Pipeline, ...]:
        """List persisted pipeline records in repository-defined order."""
        return await self._repository.list_pipelines(active_only=active_only)

    async def get(self, pipeline_id: str) -> Pipeline:
        """Retrieve one persisted pipeline by stable ID."""
        return await self._repository.get_pipeline(pipeline_id)

    async def default(self) -> Pipeline:
        """Return the active application default pipeline deterministically."""
        records = await self.list(active_only=True)
        selected = next(
            (record for record in records if record.is_default),
            next((record for record in records if record.name == "basic_rag_v1"), None),
        )
        if selected is None:
            raise KeyError("No active default pipeline is configured")
        return selected

    async def select(self, pipeline_id: str | None = None) -> Pipeline:
        """Resolve an explicit active pipeline or the configured default."""
        selected = (
            await self.default() if pipeline_id is None else await self.get(pipeline_id)
        )
        if not selected.active:
            raise ValueError(f"Pipeline {selected.id!r} is inactive")
        return selected

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

    async def update(self, pipeline_id: str, config: PipelineConfig) -> Pipeline:
        """Validate and replace a pipeline, invalidating its cached runtime."""
        self.validate(config)
        existing = await self._repository.get_pipeline_by_name(config.name)
        if existing is not None and existing.id != pipeline_id:
            raise ValueError(f"Pipeline name already exists: {config.name}")
        record = await self._repository.update_pipeline(
            pipeline_id,
            config,
            representation_hash=config.representation_hash(self._catalog),
        )
        if self._runtime is not None:
            self._runtime.invalidate(pipeline_id)
        return record

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

    @property
    def runtime(self) -> PipelineRuntimeResolver | None:
        """Return the shared resolver used by API, CLI, and chat execution."""
        return self._runtime

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
        return {
            "schema_version": 1,
            "stages": {
                "parser": self._stage(self._registry.parsers.keys(), "ComponentConfig"),
                "knowledge_builder": self._stage(
                    self._registry.knowledge_builders.keys(), "ComponentConfig"
                ),
                "chunker": self._stage(self._registry.chunkers.keys(), "ChunkerConfig"),
                "vector_backend": self._stage(
                    self._registry.vector_backends, "VectorBackendConfig"
                ),
                "retriever": self._stage(
                    self._registry.retrievers.keys(), "RetrieverConfig"
                ),
                "reranker": self._stage(
                    self._registry.rerankers.keys(), "ComponentConfig"
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
