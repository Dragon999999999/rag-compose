"""Resolve persisted pipelines into cached, executable runtime compositions."""

import asyncio
import json
from collections import OrderedDict
from dataclasses import dataclass
from hashlib import sha256
from typing import cast

from llm_grounding.domain.corpus import Pipeline
from llm_grounding.embedding.factory import build_embedder
from llm_grounding.embedding.protocol import Embedder, SparseEmbedder, TokenEmbedder
from llm_grounding.infrastructure.config.models import (
    ArchitectureConfig,
    AssessorConfig,
    ComponentConfig,
    ModelCatalog,
    PipelineConfig,
    ReasonerConfig,
    RerankerConfig,
    RetrieverConfig,
)
from llm_grounding.infrastructure.config.overrides import (
    OverrideValidationError,
    apply_query_overrides,
    catalog_for_query_overrides,
)
from llm_grounding.infrastructure.config.settings import QdrantSettings
from llm_grounding.infrastructure.database.repositories import PostgresRepository
from llm_grounding.infrastructure.model_management.manager import LocalModelManager
from llm_grounding.infrastructure.vector.learned import (
    QdrantMultiVectorStore,
    QdrantSparseVectorStore,
)
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
        local_model_manager: LocalModelManager | None = None,
    ) -> None:
        """Configure shared infrastructure and a bounded runtime cache."""
        if max_cache_size < 1:
            raise ValueError("max_cache_size must be positive")
        self._repository = repository
        self._vectors = vectors
        self._catalog = catalog
        self._registry = registry or ComponentRegistry()
        self._local_model_manager = local_model_manager
        self._max_cache_size = max_cache_size
        self._runtimes: OrderedDict[str, tuple[str, ResolvedPipeline]] = OrderedDict()
        self._embedders: OrderedDict[str, Embedder | SparseEmbedder | TokenEmbedder] = (
            OrderedDict()
        )
        self._lock = asyncio.Lock()
        self._learned_stores: dict[int, object] = {}

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

    async def close(self) -> None:
        """Close lazily created learned-index clients and clear runtime caches."""
        for store in tuple(self._learned_stores.values()):
            close = getattr(store, "close", None)
            if close is not None:
                await close()
        self._learned_stores.clear()
        self._runtimes.clear()

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
            runner = self._build_basic_rag(architecture, effective_ingestion, catalog)
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
        if architecture.renderer.implementation != "grounded_answer":
            raise PipelineResolutionError(
                "The vector runtime requires the grounded_answer renderer"
            )
        if architecture.reasoner.prompt_id is None:
            raise PipelineResolutionError("A grounded pipeline requires prompt_id")
        requirements = self._registry.query_requirements(architecture.retriever)
        dense_required = "dense_vector_index" in requirements
        sparse_required = "sparse_vector_index" in requirements
        multivector_required = "multivector_index" in requirements
        vector_store = self._vector_store_for(ingestion) if dense_required else None
        query_embedding = ingestion.embedding if dense_required else None
        if dense_required:
            if ingestion.embedding is None or ingestion.vector_backend is None:
                raise PipelineResolutionError(
                    "The selected dense query pipeline requires an indexed "
                    "embedding and vector backend"
                )
            if vector_store is None:
                raise PipelineResolutionError(
                    "The selected dense query pipeline has no vector backend available"
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
        if sparse_required or multivector_required:
            if ingestion.vector_backend != "qdrant":
                raise PipelineResolutionError(
                    "Learned sparse and multivector retrieval require the qdrant "
                    "backend because pgvector and FAISS are dense-only"
                )
            if architecture.retriever.query_embedder is None:
                raise PipelineResolutionError(
                    "The selected learned retriever requires query_embedder"
                )
        sparse_embedding = None
        sparse_embedder = None
        sparse_store = None
        token_embedder = None
        multivector_store = None
        if sparse_required:
            sparse_embedding = self._query_representation(
                architecture.retriever.query_embedder,
                catalog,
                "sparse",
            )
            sparse_embedder = self._embedder(sparse_embedding)
            if not hasattr(sparse_embedder, "embed_sparse_query"):
                raise PipelineResolutionError(
                    "Selected model does not provide sparse embeddings"
                )
            sparse_embedder = cast(SparseEmbedder, sparse_embedder)
            sparse_store = QdrantSparseVectorStore(
                QdrantSettings.from_environment(),
                self._repository,
                sparse_embedder.model_identity,
            )
            self._learned_stores[id(sparse_store)] = sparse_store
        if multivector_required:
            token_embedding = self._query_representation(
                architecture.retriever.query_embedder,
                catalog,
                "multivector",
            )
            token_embedder = self._embedder(token_embedding)
            if not hasattr(token_embedder, "embed_token_query"):
                raise PipelineResolutionError(
                    "Selected model does not provide token embeddings"
                )
            token_embedder = cast(TokenEmbedder, token_embedder)
            multivector_store = QdrantMultiVectorStore(
                QdrantSettings.from_environment(),
                self._repository,
                token_embedder.model_identity,
            )
            self._learned_stores[id(multivector_store)] = multivector_store
        dependencies = RetrieverDependencies(
            self._repository,
            vector_store,
            cast(Embedder, self._embedder(query_embedding))
            if query_embedding is not None
            else None,
            query_embedding,
            cast(SparseEmbedder, sparse_embedder),
            sparse_store,
            cast(TokenEmbedder, token_embedder),
            multivector_store,
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
        branch_retrievers = {}
        branch_rerankers = {}
        branch_assessors = {}
        for branch in architecture.execution.branches:
            if (
                branch.retriever is not None
                and branch.retriever != architecture.retriever.implementation
            ):
                branch_config = architecture.retriever.model_copy(
                    update={
                        "type": branch.retriever,
                        "implementation": branch.retriever,
                    }
                )
                branch_retrievers[branch.retriever] = self._registry.build_retriever(
                    branch_config, dependencies
                )
            if (
                branch.reranker is not None
                and branch.reranker != architecture.reranker.implementation
            ):
                branch_config = architecture.reranker.model_copy(
                    update={"implementation": branch.reranker, "type": "reranker"}
                )
                branch_rerankers[branch.reranker] = self._registry.build_reranker(
                    branch_config, self._local_model_manager
                )
            if branch.assessors:
                branch_assessors[branch.id] = tuple(
                    self._registry.build_assessor(
                        AssessorConfig(type="assessor", implementation=name),
                        self._repository,
                    )
                    for name in branch.assessors
                )
        reasoner = self._registry.build_reasoner(architecture.reasoner)
        reranker = self._registry.build_reranker(
            architecture.reranker, self._local_model_manager
        )
        alternative_retrievers = {}
        alternative_summaries = {}
        alternative_rerankers = {}
        alternative_assessors = {}
        alternative_reasoners = {}
        alternative_renderers = {}
        for alternative in architecture.execution.alternatives:
            overrides = alternative.stage_overrides
            retriever_override = overrides.get("retriever")
            if isinstance(retriever_override, dict):
                retriever_data = architecture.retriever.model_dump(mode="python")
                retriever_data.update(retriever_override)
                alternative_retriever_config = RetrieverConfig.model_validate(
                    retriever_data
                )
                alternative_retriever = self._registry.build_retriever(
                    alternative_retriever_config, dependencies
                )
                alternative_retrievers[alternative.id] = alternative_retriever
                alternative_summaries[alternative.id] = (
                    self._registry.build_summary_retriever(
                        alternative_retriever_config,
                        alternative_retriever,
                        dependencies,
                    )
                )
            reranker_override = overrides.get("reranker")
            if isinstance(reranker_override, dict):
                reranker_data = architecture.reranker.model_dump(mode="python")
                reranker_data.update(reranker_override)
                alternative_rerankers[alternative.id] = self._registry.build_reranker(
                    RerankerConfig.model_validate(reranker_data),
                    self._local_model_manager,
                )
            assessors_override = overrides.get("assessors")
            if isinstance(assessors_override, (list, tuple)):
                alternative_assessors[alternative.id] = tuple(
                    self._registry.build_assessor(
                        AssessorConfig.model_validate(item), self._repository
                    )
                    for item in assessors_override
                    if isinstance(item, dict)
                )
            reasoner_override = overrides.get("reasoner")
            if isinstance(reasoner_override, dict):
                reasoner_data = architecture.reasoner.model_dump(mode="python")
                model_ref = reasoner_override.get("model_ref")
                if isinstance(model_ref, str):
                    reasoner_data["model"] = catalog.reasoning(
                        model_ref
                    ).model_dump(mode="python")
                reasoner_data.update(
                    {
                        key: value
                        for key, value in reasoner_override.items()
                        if key != "model_ref"
                    }
                )
                alternative_reasoners[alternative.id] = self._registry.build_reasoner(
                    ReasonerConfig.model_validate(reasoner_data)
                )
            renderer_override = overrides.get("renderer")
            if isinstance(renderer_override, dict):
                renderer_data = architecture.renderer.model_dump(mode="python")
                renderer_data.update(renderer_override)
                alternative_renderers[alternative.id] = self._registry.build_renderer(
                    ComponentConfig.model_validate(renderer_data)
                )
            for branch in alternative.branches:
                if (
                    branch.retriever is not None
                    and branch.retriever not in branch_retrievers
                ):
                    branch_config = architecture.retriever.model_copy(
                        update={"implementation": branch.retriever}
                    )
                    branch_retrievers[branch.retriever] = (
                        self._registry.build_retriever(branch_config, dependencies)
                    )
                if (
                    branch.reranker is not None
                    and branch.reranker not in branch_rerankers
                ):
                    branch_config = architecture.reranker.model_copy(
                        update={"implementation": branch.reranker}
                    )
                    branch_rerankers[branch.reranker] = (
                        self._registry.build_reranker(
                            branch_config, self._local_model_manager
                        )
                    )
                if branch.assessors:
                    branch_assessors[f"{alternative.id}:{branch.id}"] = tuple(
                        self._registry.build_assessor(
                            AssessorConfig(type="assessor", implementation=name),
                            self._repository,
                        )
                        for name in branch.assessors
                    )
        return BasicRagRunner(
            retriever,
            summary,
            reasoner,
            tuple(
                self._registry.build_assessor(item, self._repository)
                for item in architecture.assessors
            ),
            self._registry.build_decision_policy(architecture.decision_policy),
            self._registry.build_renderer(architecture.renderer),
            embedding_config=query_embedding,
            reranker=reranker,
            reranker_config=architecture.reranker,
            reranker_builder=lambda config: self._registry.build_reranker(
                config, self._local_model_manager
            ),
            repository=self._repository,
            execution=architecture.execution,
            branch_retrievers=branch_retrievers,
            branch_rerankers=branch_rerankers,
            branch_assessors=branch_assessors,
            alternative_retrievers=alternative_retrievers,
            alternative_summaries=alternative_summaries,
            alternative_rerankers=alternative_rerankers,
            alternative_assessors=alternative_assessors,
            alternative_reasoners=alternative_reasoners,
            alternative_renderers=alternative_renderers,
        )

    def _vector_store_for(self, ingestion: IngestionConfig) -> VectorStore | None:
        """Resolve a project pipeline's backend through the application router."""
        if self._vectors is None:
            return None
        resolver = getattr(self._vectors, "for_ingestion_config", None)
        return resolver(ingestion) if resolver is not None else self._vectors

    def _embedder(
        self, config: EmbeddingConfig
    ) -> Embedder | SparseEmbedder | TokenEmbedder:
        """Reuse embedding adapters by representation identity with LRU bounds."""
        identity = config.identity()
        cached = self._embedders.get(identity)
        if cached is not None:
            self._embedders.move_to_end(identity)
            return cached
        embedder = build_embedder(config, self._local_model_manager)
        self._embedders[identity] = embedder
        self._embedders.move_to_end(identity)
        while len(self._embedders) > self._max_cache_size:
            self._embedders.popitem(last=False)
        return embedder

    @staticmethod
    def _query_representation(
        model_ref: str | None,
        catalog: ModelCatalog,
        representation: str,
    ) -> EmbeddingConfig:
        """Resolve a query model and adapt shared BGE weights to its representation."""
        if model_ref is None:
            raise PipelineResolutionError("A query model reference is required")
        try:
            config = catalog.embedding(model_ref)
        except ValueError as exc:
            raise PipelineResolutionError(str(exc)) from exc
        if config.representation == representation:
            return config
        if config.model_id == "embedding_bge_m3" and representation == "sparse":
            return config.model_copy(update={"representation": "sparse"})
        if representation == "multivector":
            return config.model_copy(update={"representation": "multivector"})
        raise PipelineResolutionError(
            f"Model {model_ref!r} does not declare {representation} support"
        )

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
