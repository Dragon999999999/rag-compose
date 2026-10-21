"""Resolve persisted pipelines into cached, executable runtime compositions."""

import asyncio
import json
from collections import OrderedDict
from dataclasses import dataclass
from hashlib import sha256

from llm_grounding.assessment.source_reference import SourceReferenceAssessor
from llm_grounding.decision.basic_grounding import BasicGroundingPolicy
from llm_grounding.domain.corpus import Pipeline
from llm_grounding.embedding.factory import build_embedder
from llm_grounding.embedding.protocol import Embedder
from llm_grounding.infrastructure.config.models import (
    ArchitectureConfig,
    ModelCatalog,
    PipelineConfig,
)
from llm_grounding.infrastructure.database.repositories import PostgresRepository
from llm_grounding.infrastructure.llm.aqueduct import AqueductModelClient
from llm_grounding.infrastructure.prompts import PromptLibrary
from llm_grounding.infrastructure.vector.protocol import VectorStore
from llm_grounding.ingestion.models import IngestionConfig
from llm_grounding.orchestration.basic_rag import BasicRagRunner
from llm_grounding.orchestration.registry import ComponentRegistry
from llm_grounding.orchestration.runner import ArchitectureRunner
from llm_grounding.reasoning.basic_grounded import BasicGroundedReasoner
from llm_grounding.rendering.grounded_answer import GroundedAnswerRenderer
from llm_grounding.retrieval.vector import SummaryRetriever, VectorRetriever


class PipelineResolutionError(ValueError):
    """Report a stored pipeline that cannot be turned into a runnable composition."""


@dataclass(frozen=True)
class ResolvedPipeline:
    """Pair a persisted pipeline with its validated runtime configurations."""

    record: Pipeline
    config: PipelineConfig
    architecture: ArchitectureConfig
    ingestion: IngestionConfig
    runner: BasicRagRunner | ArchitectureRunner


class PipelineRuntimeResolver:
    """Resolve pipeline IDs with bounded LRU caching and hash invalidation.

    The resolver owns no database or vector-store lifecycle. It reuses the
    application-owned infrastructure clients and only caches provider adapters,
    runners, and resolved configuration for active pipeline records.
    """

    def __init__(
        self,
        repository: PostgresRepository,
        vectors: VectorStore,
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
        self, pipeline_id: str, *, ingestion: IngestionConfig | None = None
    ) -> ResolvedPipeline:
        """Load, validate, and construct a query or complete pipeline runtime."""
        record = await self._repository.get_pipeline(pipeline_id)
        if not record.active:
            raise PipelineResolutionError(f"Pipeline {pipeline_id!r} is inactive")
        config = self._parse_config(record)
        if config.kind == "query" and ingestion is None:
            raise PipelineResolutionError(
                "A query pipeline requires the project's effective ingestion config"
            )
        signature = self._signature(record, config, ingestion)
        async with self._lock:
            cached = self._runtimes.get(pipeline_id)
            if cached is not None and cached[0] == signature:
                self._runtimes.move_to_end(pipeline_id)
                return cached[1]
            runtime = self._build(record, config, ingestion)
            self._runtimes[pipeline_id] = (signature, runtime)
            self._runtimes.move_to_end(pipeline_id)
            while len(self._runtimes) > self._max_cache_size:
                self._runtimes.popitem(last=False)
            return runtime

    async def resolve_pipeline(
        self, pipeline_id: str, *, ingestion: IngestionConfig | None = None
    ) -> ResolvedPipeline:
        """Descriptive alias for ``resolve`` used by external application clients."""
        return await self.resolve(pipeline_id, ingestion=ingestion)

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
            self._registry.validate_pipeline(config, self._catalog)
            expected_representation = config.representation_hash(self._catalog)
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
    ) -> ResolvedPipeline:
        """Build a runtime while retaining existing execution engines."""
        try:
            if config.kind == "full":
                architecture, effective_ingestion = config.resolve(self._catalog)
            elif config.kind == "query":
                if ingestion is None:
                    raise PipelineResolutionError(
                        "A query pipeline requires a project index configuration"
                    )
                architecture, effective_ingestion = config.resolve_query(
                    self._catalog, ingestion
                )
            else:
                raise PipelineResolutionError(
                    "Ingestion pipelines are not executable query runtimes"
                )
            if architecture.retriever.type == "vector":
                runner = self._build_basic_rag(architecture, effective_ingestion)
            else:
                runner = ArchitectureRunner(self._registry.build(architecture))
            return ResolvedPipeline(
                record, config, architecture, effective_ingestion, runner
            )
        except PipelineResolutionError:
            raise
        except (RuntimeError, ValueError, TypeError) as exc:
            raise PipelineResolutionError(
                f"Pipeline {record.id!r} could not be constructed"
            ) from exc

    def _build_basic_rag(
        self,
        architecture: ArchitectureConfig,
        ingestion: IngestionConfig,
    ) -> BasicRagRunner:
        """Construct the supported vector-grounded runner from selected stages."""
        if architecture.reasoner.type != "basic_grounded":
            raise PipelineResolutionError(
                "The vector runtime requires the basic_grounded reasoner"
            )
        if architecture.reranker.type != "identity":
            raise PipelineResolutionError(
                "The vector runtime currently supports only the identity reranker"
            )
        if tuple(item.type for item in architecture.assessors) != ("source_reference",):
            raise PipelineResolutionError(
                "The vector runtime requires the source_reference assessor"
            )
        if architecture.decision_policy.type != "basic_grounding":
            raise PipelineResolutionError(
                "The vector runtime requires the basic_grounding decision policy"
            )
        if architecture.renderer.type != "grounded_answer":
            raise PipelineResolutionError(
                "The vector runtime requires the grounded_answer renderer"
            )
        if architecture.reasoner.prompt_id is None:
            raise PipelineResolutionError("A grounded pipeline requires prompt_id")
        embedder = self._embedder(ingestion)
        vector = VectorRetriever(
            embedder,
            self._vectors,
            architecture.retriever.top_k or 8,
            embedding_config=ingestion.embedding,
        )
        summary = SummaryRetriever(
            self._repository,
            vector,
            architecture.retriever.context_chars or 12000,
            architecture.retriever.summary_per_facet or 3,
        )
        reasoner = BasicGroundedReasoner(
            AqueductModelClient(architecture.reasoner.model),
            PromptLibrary.default(),
            architecture.reasoner.prompt_id,
        )
        return BasicRagRunner(
            vector,
            summary,
            reasoner,
            SourceReferenceAssessor(),
            BasicGroundingPolicy(),
            GroundedAnswerRenderer(),
            embedding_config=ingestion.embedding,
        )

    def _embedder(self, ingestion: IngestionConfig) -> Embedder:
        """Reuse embedding adapters by representation identity with LRU bounds."""
        identity = ingestion.embedding.identity()
        cached = self._embedders.get(identity)
        if cached is not None:
            self._embedders.move_to_end(identity)
            return cached
        embedder = build_embedder(ingestion.embedding)
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
    ) -> str:
        """Include persisted and catalog-resolved settings in cache invalidation."""
        catalog_data = self._catalog.model_dump(mode="json")
        canonical = json.dumps(
            {
                "config_hash": record.config_hash,
                "representation_hash": record.representation_hash,
                "catalog": catalog_data,
                "effective_representation": config.representation_hash(self._catalog),
                "project_ingestion": (
                    ingestion.identity() if ingestion is not None else None
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return sha256(canonical.encode()).hexdigest()
