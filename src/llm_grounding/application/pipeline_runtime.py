"""Resolve persisted pipelines into cached, executable runtime compositions."""

import asyncio
import json
from collections import OrderedDict
from dataclasses import dataclass
from hashlib import sha256

from llm_grounding.domain.corpus import Pipeline
from llm_grounding.embedding.factory import build_embedder
from llm_grounding.embedding.protocol import Embedder
from llm_grounding.infrastructure.config.models import (
    ArchitectureConfig,
    ModelCatalog,
    PipelineConfig,
)
from llm_grounding.infrastructure.config.overrides import (
    OverrideValidationError,
    apply_query_overrides,
    catalog_for_query_overrides,
)
from llm_grounding.infrastructure.database.repositories import PostgresRepository
from llm_grounding.infrastructure.vector.protocol import VectorStore
from llm_grounding.ingestion.models import EmbeddingConfig, IngestionConfig
from llm_grounding.orchestration.basic_rag import BasicRagRunner
from llm_grounding.orchestration.registry import (
    ComponentRegistry,
    RetrieverDependencies,
)


class PipelineResolutionError(ValueError):
    """Report a stored pipeline that cannot be turned into a runnable composition."""


@dataclass(frozen=True)
class ResolvedPipeline:
    """Pair a persisted pipeline with its validated runtime configurations."""

    record: Pipeline
    config: PipelineConfig
    override_provenance: dict[str, object]
    architecture: ArchitectureConfig
    ingestion: IngestionConfig
    runner: BasicRagRunner


class PipelineRuntimeResolver:
    """Resolve pipeline IDs with bounded LRU caching and hash invalidation.

    The resolver owns no database or vector-store lifecycle. It reuses the
    application-owned infrastructure clients and only caches provider adapters,
    runners, and resolved configuration for active pipeline records.
    """

    def __init__(
        self,
        repository: PostgresRepository,
        vectors: VectorStore | None,
        catalog: ModelCatalog,
        *,
        max_cache_size: int = 8,
        registry: ComponentRegistry | None = None,
    ) -> None:
        """Configure shared infrastructure and a bounded runtime cache."""
        if max_cache_size < 1:
            raise ValueError("max_cache_size must be positive")
        self._repository = repository
        self._vectors = vectors
        self._catalog = catalog
        self._registry = registry or ComponentRegistry()
        self._max_cache_size = max_cache_size
        self._runtimes: OrderedDict[str, tuple[str, ResolvedPipeline]] = OrderedDict()
        self._embedders: OrderedDict[str, Embedder] = OrderedDict()
        self._lock = asyncio.Lock()

    async def resolve(
        self,
        pipeline_id: str,
        *,
        ingestion: IngestionConfig | None = None,
        chat_overrides: dict[str, dict[str, object]] | None = None,
        request_overrides: dict[str, dict[str, object]] | None = None,
    ) -> ResolvedPipeline:
        """Load, validate, and construct a query pipeline runtime."""
        record = await self._repository.get_pipeline(pipeline_id)
        if not record.active:
            raise PipelineResolutionError(f"Pipeline {pipeline_id!r} is inactive")
        config = self._parse_config(record)
        if config.kind == "query" and ingestion is None:
            raise PipelineResolutionError(
                "A query pipeline requires the project's effective ingestion config"
            )
        signature = self._signature(
            record,
            config,
            ingestion,
            chat_overrides,
            request_overrides,
        )
        async with self._lock:
            cached = self._runtimes.get(pipeline_id)
            if cached is not None and cached[0] == signature:
                self._runtimes.move_to_end(pipeline_id)
                return cached[1]
            runtime = self._build(
                record,
                config,
                ingestion,
                chat_overrides=chat_overrides,
                request_overrides=request_overrides,
            )
            self._runtimes[pipeline_id] = (signature, runtime)
            self._runtimes.move_to_end(pipeline_id)
            while len(self._runtimes) > self._max_cache_size:
                self._runtimes.popitem(last=False)
            return runtime

    async def resolve_pipeline(
        self,
        pipeline_id: str,
        *,
        ingestion: IngestionConfig | None = None,
        chat_overrides: dict[str, dict[str, object]] | None = None,
        request_overrides: dict[str, dict[str, object]] | None = None,
    ) -> ResolvedPipeline:
        """Descriptive alias for ``resolve`` used by external application clients."""
        return await self.resolve(
            pipeline_id,
            ingestion=ingestion,
            chat_overrides=chat_overrides,
            request_overrides=request_overrides,
        )

    def invalidate(self, pipeline_id: str | None = None) -> None:
        """Drop one runtime or all runtimes after a configuration mutation."""
        if pipeline_id is None:
            self._runtimes.clear()
        else:
            self._runtimes.pop(pipeline_id, None)

    @property
    def cache_size(self) -> int:
        """Return the number of cached pipeline runtime instances."""
        return len(self._runtimes)

    def _parse_config(self, record: Pipeline) -> PipelineConfig:
        """Validate persisted JSON and its closed component/model references."""
        try:
            config = PipelineConfig.model_validate(record.config)
            catalog = config.effective_catalog(self._catalog)
            self._registry.validate_pipeline(config, catalog)
            expected_representation = config.representation_hash(catalog)
            if record.representation_hash != expected_representation:
                raise PipelineResolutionError(
                    f"Pipeline {record.id!r} representation hash does not match "
                    "its model catalog"
                )
            return config
        except PipelineResolutionError:
            raise
        except (ValueError, TypeError) as exc:
            raise PipelineResolutionError(
                f"Stored pipeline {record.id!r} is not runnable"
            ) from exc

    def _build(
        self,
        record: Pipeline,
        config: PipelineConfig,
        ingestion: IngestionConfig | None,
        *,
        chat_overrides: dict[str, dict[str, object]] | None,
        request_overrides: dict[str, dict[str, object]] | None,
    ) -> ResolvedPipeline:
        """Build a runtime while retaining existing execution engines."""
        try:
            if config.kind != "query" or ingestion is None:
                raise PipelineResolutionError(
                    "A query pipeline requires the project's index configuration"
                )
            override_layers = (
                ("pipeline", config.stage_overrides),
                ("chat", chat_overrides or {}),
                ("request", request_overrides or {}),
            )
            catalog = catalog_for_query_overrides(
                config, self._catalog, override_layers
            )
            effective_config, provenance = apply_query_overrides(
                config, override_layers
            )
            self._registry.validate_pipeline(effective_config, catalog)
            architecture, effective_ingestion = effective_config.resolve_query(
                catalog, ingestion
            )
            runner = self._build_basic_rag(
                architecture, effective_ingestion, catalog
            )
            return ResolvedPipeline(
                record,
                effective_config,
                provenance,
                architecture,
                effective_ingestion,
                runner,
            )
        except PipelineResolutionError:
            raise
        except OverrideValidationError as exc:
            raise PipelineResolutionError(str(exc)) from exc
        except (RuntimeError, ValueError, TypeError) as exc:
            raise PipelineResolutionError(
                f"Pipeline {record.id!r} could not be constructed"
            ) from exc

    def _build_basic_rag(
        self,
        architecture: ArchitectureConfig,
        ingestion: IngestionConfig,
        catalog: ModelCatalog,
    ) -> BasicRagRunner:
        """Construct the selected retriever with only its required resources."""
        if architecture.reasoner.implementation != "basic_grounded":
            raise PipelineResolutionError(
                "The vector runtime requires the basic_grounded reasoner"
            )
        if tuple(item.implementation for item in architecture.assessors) != (
            "source_reference",
        ):
            raise PipelineResolutionError(
                "The vector runtime requires the source_reference assessor"
            )
        if architecture.decision_policy.implementation != "basic_grounding":
            raise PipelineResolutionError(
                "The vector runtime requires the basic_grounding decision policy"
            )
        if architecture.renderer.implementation != "grounded_answer":
            raise PipelineResolutionError(
                "The vector runtime requires the grounded_answer renderer"
            )
        if architecture.reasoner.prompt_id is None:
            raise PipelineResolutionError("A grounded pipeline requires prompt_id")
        requirements = self._registry.query_requirements(architecture.retriever)
        dense_required = "dense_vector_index" in requirements
        query_embedding = ingestion.embedding if dense_required else None
        if dense_required:
            if ingestion.embedding is None or ingestion.vector_backend is None:
                raise PipelineResolutionError(
                    "The selected dense query pipeline requires an indexed "
                    "embedding and vector backend"
                )
            if self._vectors is None:
                raise PipelineResolutionError(
                    "The selected dense query pipeline has no vector backend "
                    "available"
                )
            if architecture.retriever.query_embedder is not None:
                try:
                    query_embedding = catalog.embedding(
                        architecture.retriever.query_embedder
                    )
                except ValueError as exc:
                    raise PipelineResolutionError(str(exc)) from exc
                if not query_embedding.is_compatible_with(ingestion.embedding):
                    raise PipelineResolutionError(
                        "The selected query embedder is incompatible with the "
                        "indexed embedding space"
                    )
            if query_embedding is None:
                raise PipelineResolutionError(
                    "Dense query runtime requires an embedding provider"
                )
        dependencies = RetrieverDependencies(
            self._repository,
            self._vectors if dense_required else None,
            self._embedder(query_embedding) if query_embedding is not None else None,
            query_embedding,
        )
        try:
            retriever = self._registry.build_retriever(
                architecture.retriever, dependencies
            )
            summary = self._registry.build_summary_retriever(
                architecture.retriever, retriever, dependencies
            )
        except (RuntimeError, ValueError, TypeError) as exc:
            raise PipelineResolutionError(
                "The selected retriever could not be constructed"
            ) from exc
        reasoner = self._registry.build_reasoner(architecture.reasoner)
        reranker = self._registry.build_reranker(architecture.reranker)
        return BasicRagRunner(
            retriever,
            summary,
            reasoner,
            self._registry.build_assessor(architecture.assessors[0]),
            self._registry.build_decision_policy(architecture.decision_policy),
            self._registry.build_renderer(architecture.renderer),
            embedding_config=query_embedding,
            reranker=reranker,
            reranker_config=architecture.reranker,
            reranker_builder=self._registry.build_reranker,
            repository=self._repository,
        )

    def _embedder(self, config: EmbeddingConfig) -> Embedder:
        """Reuse embedding adapters by representation identity with LRU bounds."""
        identity = config.identity()
        cached = self._embedders.get(identity)
        if cached is not None:
            self._embedders.move_to_end(identity)
            return cached
        embedder = build_embedder(config)
        self._embedders[identity] = embedder
        self._embedders.move_to_end(identity)
        while len(self._embedders) > self._max_cache_size:
            self._embedders.popitem(last=False)
        return embedder

    def _signature(
        self,
        record: Pipeline,
        config: PipelineConfig,
        ingestion: IngestionConfig | None,
        chat_overrides: dict[str, dict[str, object]] | None = None,
        request_overrides: dict[str, dict[str, object]] | None = None,
    ) -> str:
        """Include persisted and catalog-resolved settings in cache invalidation."""
        catalog_data = config.effective_catalog(self._catalog).model_dump(mode="json")
        canonical = json.dumps(
            {
                "config_hash": record.config_hash,
                "representation_hash": record.representation_hash,
                "catalog": catalog_data,
                "effective_representation": config.representation_hash(self._catalog),
                "project_ingestion": (
                    ingestion.identity() if ingestion is not None else None
                ),
                "chat_overrides": chat_overrides or {},
                "request_overrides": request_overrides or {},
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return sha256(canonical.encode()).hexdigest()
