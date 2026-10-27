"""Small typed factory registry for configured components, without dynamic plugins.

Validate every identifier before constructing anything. Alternative implementations
can be added to the corresponding typed mapping; composition never branches on
architecture names. A supplied model client replaces external generation for tests.
"""

from collections.abc import Callable, Collection
from dataclasses import dataclass

from llm_grounding.assessment.source_reference import SourceReferenceAssessor
from llm_grounding.decision.basic_grounding import BasicGroundingPolicy
from llm_grounding.embedding.protocol import Embedder
from llm_grounding.infrastructure.config.loader import ConfigurationError
from llm_grounding.infrastructure.config.models import (
    ComponentConfig,
    ModelCatalog,
    PipelineConfig,
    ReasonerConfig,
    RerankerConfig,
    RetrieverConfig,
)
from llm_grounding.infrastructure.database.repositories import PostgresRepository
from llm_grounding.infrastructure.llm.aqueduct import AqueductModelClient
from llm_grounding.infrastructure.prompts import PromptLibrary, PromptLibraryError
from llm_grounding.infrastructure.vector.protocol import VectorStore
from llm_grounding.ingestion.models import ChunkerConfig, EmbeddingConfig
from llm_grounding.knowledge.basic import BasicKnowledgeBuilder
from llm_grounding.knowledge.chunking.block_aware import BlockAwareChunker
from llm_grounding.knowledge.chunking.fixed_window import FixedWindowChunker
from llm_grounding.knowledge.chunking.protocol import Chunker
from llm_grounding.knowledge.protocol import KnowledgeBuilder
from llm_grounding.parsing.protocol import DocumentParser
from llm_grounding.parsing.pymupdf import PyMuPDFDocumentParser
from llm_grounding.reasoning.basic_grounded import BasicGroundedReasoner
from llm_grounding.rendering.grounded_answer import GroundedAnswerRenderer
from llm_grounding.retrieval.full_context import FullContextRetriever
from llm_grounding.retrieval.hybrid import HybridRetriever
from llm_grounding.retrieval.protocol import Retriever
from llm_grounding.retrieval.reranking.cross_encoder import CrossEncoderReranker
from llm_grounding.retrieval.reranking.identity import IdentityReranker
from llm_grounding.retrieval.reranking.protocol import Reranker
from llm_grounding.retrieval.reranking.top_k import TopKReranker
from llm_grounding.retrieval.sparse import SparseRetriever
from llm_grounding.retrieval.vector import SummaryRetriever, VectorRetriever


@dataclass(frozen=True)
class RetrieverDependencies:
    """Resolved resources available to a project-scoped retriever factory."""

    repository: PostgresRepository
    vector_store: VectorStore | None = None
    embedder: Embedder | None = None
    embedding_config: EmbeddingConfig | None = None


def _vector_retriever_requires_runtime_dependencies() -> Retriever:
    """Reserve the registered vector stage for the Qdrant composition root."""
    raise ConfigurationError(
        "The vector retriever requires the Qdrant repository and embedder; "
        "construct it through the application bootstrap."
    )


def _retriever_requires_runtime_dependencies(name: str) -> Retriever:
    """Reserve a retriever name for construction by the application composition root."""
    raise ConfigurationError(
        f"The {name} retriever requires project-scoped runtime dependencies; "
        "construct it through the application bootstrap."
    )


class ComponentRegistry:
    """Map stage-local names to typed constructors and validate complete composition."""

    def __init__(self) -> None:
        """Register the supported split-pipeline implementations."""
        self.parsers: dict[str, Callable[[], DocumentParser]] = {
            "pymupdf": PyMuPDFDocumentParser
        }
        self.knowledge_builders: dict[str, Callable[[], KnowledgeBuilder]] = {
            "basic": BasicKnowledgeBuilder
        }
        self.retrievers: dict[str, Callable[[], Retriever]] = {
            "vector_similarity": _vector_retriever_requires_runtime_dependencies,
            "postgres_all_chunks": lambda: _retriever_requires_runtime_dependencies(
                "postgres_all_chunks"
            ),
            "postgres_fts": lambda: _retriever_requires_runtime_dependencies(
                "postgres_fts"
            ),
            "hybrid_rrf": lambda: _retriever_requires_runtime_dependencies(
                "hybrid_rrf"
            ),
        }
        self.rerankers: dict[str, Callable[[RerankerConfig], Reranker]] = {
            "identity": lambda config: IdentityReranker(),
            "top_k": lambda config: TopKReranker(config.rerank_top_k),
            "cross_encoder": lambda config: CrossEncoderReranker(
                model_name=config.model_name,
                device=config.device,
                batch_size=config.batch_size,
                top_k=config.rerank_top_k,
            ),
        }
        self.pipeline_reasoners = frozenset({"basic_grounded"})
        self.pipeline_assessors = frozenset({"source_reference"})
        self.pipeline_decision_policies = frozenset({"basic_grounding"})
        self.pipeline_renderers = frozenset({"grounded_answer"})
        self.chunkers: dict[str, Callable[[ChunkerConfig], Chunker]] = {
            "block_aware": BlockAwareChunker,
            "fixed_window": FixedWindowChunker,
        }
        self._retriever_factories: dict[
            str, Callable[[RetrieverConfig, RetrieverDependencies], Retriever]
        ] = {
            "vector_similarity": self._build_vector_retriever,
            "postgres_all_chunks": self._build_full_context_retriever,
            "postgres_fts": self._build_sparse_retriever,
            "hybrid_rrf": self._build_hybrid_retriever,
        }
        self.vector_backends: frozenset[str] = frozenset(
            {"qdrant", "pgvector", "faiss"}
        )

    def build_parser(self, config: ComponentConfig) -> DocumentParser:
        """Construct the configured parser from the closed parser registry."""
        try:
            factory = self.parsers[config.implementation or config.type]
        except KeyError as exc:
            raise ConfigurationError(
                f"Unknown parser implementation {config.implementation!r}"
            ) from exc
        return factory()

    def build_chunker(self, config: ChunkerConfig) -> Chunker:
        """Construct the configured chunker from validated operational settings."""
        try:
            factory = self.chunkers[config.implementation or config.type]
        except KeyError as exc:
            raise ConfigurationError(
                f"Unknown chunker implementation {config.implementation!r}"
            ) from exc
        return factory(config)

    def build_retriever(
        self,
        config: RetrieverConfig,
        dependencies: RetrieverDependencies,
    ) -> Retriever:
        """Construct one retriever using only resources declared by its contract."""
        try:
            factory = self._retriever_factories[config.implementation or config.type]
        except KeyError as exc:
            raise ConfigurationError(
                f"Unknown retriever implementation {config.implementation!r}"
            ) from exc
        return factory(config, dependencies)

    def build_summary_retriever(
        self,
        config: RetrieverConfig,
        retriever: Retriever,
        dependencies: RetrieverDependencies,
    ) -> Retriever:
        """Construct the dense summary adapter while preserving other strategies."""
        if config.type != "dense":
            return retriever
        if not isinstance(retriever, VectorRetriever):
            raise ConfigurationError(
                "The vector summary adapter requires a VectorRetriever instance"
            )
        if config.context_chars is None and config.summary_per_facet is None:
            return retriever
        return SummaryRetriever(
            dependencies.repository,
            retriever,
            config.context_chars or 12000,
            config.summary_per_facet or 3,
        )

    def build_reasoner(self, config: ReasonerConfig) -> BasicGroundedReasoner:
        """Construct the configured reasoning engine through its stage registry."""
        if config.implementation != "basic_grounded":
            raise ConfigurationError(
                f"Unknown reasoner component {config.type!r}"
            )
        if config.prompt_id is None:
            raise ConfigurationError("The basic_grounded reasoner requires prompt_id")
        return BasicGroundedReasoner(
            AqueductModelClient(config.model),
            PromptLibrary.default(),
            config.prompt_id,
        )

    def build_assessor(self, config: ComponentConfig) -> SourceReferenceAssessor:
        """Construct one registered evidence assessor."""
        if config.implementation != "source_reference":
            raise ConfigurationError(f"Unknown assessor component {config.type!r}")
        return SourceReferenceAssessor()

    def build_decision_policy(self, config: ComponentConfig) -> BasicGroundingPolicy:
        """Construct one registered decision policy."""
        if config.implementation != "basic_grounding":
            raise ConfigurationError(
                f"Unknown decision_policy component {config.type!r}"
            )
        return BasicGroundingPolicy()

    def build_renderer(self, config: ComponentConfig) -> GroundedAnswerRenderer:
        """Construct one registered answer renderer."""
        if config.implementation != "grounded_answer":
            raise ConfigurationError(f"Unknown renderer component {config.type!r}")
        return GroundedAnswerRenderer()

    def validate_pipeline(
        self, config: PipelineConfig, catalog: ModelCatalog | None = None
    ) -> None:
        """Validate all pipeline stages and named model references.

        The registry remains closed: a pipeline can choose only implementations
        registered here and can never name an arbitrary import path.

        Args:
            config: Complete pipeline composition to validate.
            catalog: Shared catalog used to resolve embedding and reasoning models.

        Raises:
            ConfigurationError: If a component, model reference, or stage pairing
                is unsupported.
        """
        selections: list[tuple[str, str, Collection[str]]] = []
        if config.representation is not None:
            selections.append(
                (
                    "parser",
                    config.representation.parser.implementation
                    or config.representation.parser.type,
                    self.parsers,
                )
            )
            if config.representation.knowledge_builder is not None:
                selections.append(
                    (
                        "knowledge_builder",
                        config.representation.knowledge_builder.implementation
                        or config.representation.knowledge_builder.type,
                        self.knowledge_builders,
                    )
                )
        if config.query is not None:
            selections.extend(
                (
                    (
                        "retriever",
                        config.query.retriever.implementation
                        or config.query.retriever.type,
                        self.retrievers,
                    ),
                    (
                        "reranker",
                        config.query.reranker.implementation,
                        self.rerankers,
                    ),
                    (
                        "reasoner",
                        config.query.reasoner.implementation
                        or config.query.reasoner.type,
                        self.pipeline_reasoners,
                    ),
                    (
                        "decision_policy",
                        config.query.decision_policy.implementation
                        or config.query.decision_policy.type,
                        self.pipeline_decision_policies,
                    ),
                    (
                        "renderer",
                        config.query.renderer.implementation
                        or config.query.renderer.type,
                        self.pipeline_renderers,
                    ),
                )
            )
            selections.extend(
                (
                    "assessor",
                    item.implementation or item.type,
                    self.pipeline_assessors,
                )
                for item in config.query.assessors
            )
        for stage, name, factories in selections:
            if name not in factories:
                raise ConfigurationError(
                    f"Unknown {stage} component {name!r}; "
                    f"available: {', '.join(factories)}"
                )
        if (
            config.representation is not None
            and (
                config.representation.chunker.implementation
                or config.representation.chunker.type
            )
            not in self.chunkers
        ):
            raise ConfigurationError(
                "Unknown chunker implementation "
                f"{config.representation.chunker.implementation!r}; "
                f"available: {', '.join(self.chunkers)}"
            )
        if (
            config.representation is not None
            and config.representation.vector_backend is not None
            and config.representation.vector_backend.implementation
            not in self.vector_backends
        ):
            raise ConfigurationError(
                "Unknown vector_backend component "
                f"{config.representation.vector_backend.implementation!r}; available: "
                f"{', '.join(sorted(self.vector_backends))}"
            )
        if config.representation is not None:
            if config.representation.embedding is None and (
                config.representation.vector_backend is not None
            ):
                raise ConfigurationError(
                    "vector_backend requires an embedding configuration"
                )
            if config.representation.embedding is not None and (
                config.representation.vector_backend is None
            ):
                raise ConfigurationError(
                    "An embedding configuration requires a resolved vector_backend"
                )
        if catalog is not None:
            try:
                if config.representation is not None:
                    if config.representation.embedding is not None:
                        catalog.embedding(config.representation.embedding.model_ref)
                if config.query is not None and config.query.retriever.query_embedder:
                    catalog.embedding(config.query.retriever.query_embedder)
                if config.query is not None:
                    catalog.reasoning(config.query.reasoner.model_ref)
            except ValueError as exc:
                kind = "embedding" if "embedding" in str(exc) else "reasoning"
                raise ConfigurationError(f"Unknown {kind} model reference") from exc
        if (
            config.query is not None
            and config.query.retriever.type in {"dense", "hybrid"}
            and (
                config.representation is not None
                and config.representation.vector_backend is not None
                and config.representation.vector_backend.implementation != "qdrant"
            )
        ):
            raise ConfigurationError(
                "The vector retriever requires the qdrant vector backend"
            )
        if config.query is not None and (
            config.query.reasoner.implementation == "basic_grounded"
            and not config.query.reasoner.prompt_id
        ):
            raise ConfigurationError(
                "The basic_grounded reasoner requires a registered prompt_id"
            )
        if config.query is not None and config.query.reasoner.prompt_id is not None:
            try:
                PromptLibrary.default().metadata(config.query.reasoner.prompt_id)
            except PromptLibraryError as exc:
                raise ConfigurationError(str(exc)) from exc

    @staticmethod
    def _build_vector_retriever(
        config: RetrieverConfig, dependencies: RetrieverDependencies
    ) -> Retriever:
        """Build dense retrieval after its vector and embedding resources resolve."""
        if dependencies.vector_store is None or dependencies.embedder is None:
            raise ConfigurationError(
                "The dense retriever requires a vector backend and embedder"
            )
        return VectorRetriever(
            dependencies.embedder,
            dependencies.vector_store,
            config.top_k or config.dense_top_k or 8,
            embedding_config=dependencies.embedding_config,
        )

    @staticmethod
    def _build_full_context_retriever(
        config: RetrieverConfig, dependencies: RetrieverDependencies
    ) -> Retriever:
        """Build full-context retrieval from canonical PostgreSQL chunks only."""
        return FullContextRetriever(
            dependencies.repository,
            config.context_budget_tokens or 16_000,
        )

    @staticmethod
    def _build_sparse_retriever(
        config: RetrieverConfig, dependencies: RetrieverDependencies
    ) -> Retriever:
        """Build PostgreSQL FTS retrieval without dense resources."""
        return SparseRetriever(dependencies.repository, config.top_k or 20)

    def _build_hybrid_retriever(
        self, config: RetrieverConfig, dependencies: RetrieverDependencies
    ) -> Retriever:
        """Compose registered dense and sparse factories with weighted RRF settings."""
        dense_config = RetrieverConfig(
            type="vector",
            top_k=config.dense_top_k or 20,
            requirements=config.requirements,
        )
        sparse_config = RetrieverConfig(
            type="postgres_fts",
            top_k=config.sparse_top_k or 20,
            requirements=config.requirements,
        )
        dense = self.build_retriever(dense_config, dependencies)
        sparse = self.build_retriever(sparse_config, dependencies)
        return HybridRetriever(
            dense,
            sparse,
            dense_top_k=config.dense_top_k or 20,
            sparse_top_k=config.sparse_top_k or 20,
            top_k=config.top_k or 10,
            rrf_k=config.rrf_k,
            dense_weight=config.dense_weight,
            sparse_weight=config.sparse_weight,
        )

    def query_capabilities(self, retriever_type: str) -> frozenset[str]:
        """Return resource capabilities for a registered retriever implementation."""
        if retriever_type == "vector_similarity":
            return frozenset({"chunk_vectors", "dense_vector_index"})
        if retriever_type == "postgres_all_chunks":
            return frozenset({"persisted_chunks"})
        if retriever_type == "postgres_fts":
            return frozenset({"persisted_chunks", "postgres_fts"})
        if retriever_type == "hybrid_rrf":
            return frozenset(
                {
                    "persisted_chunks",
                    "postgres_fts",
                    "chunk_vectors",
                    "dense_vector_index",
                }
            )
        raise ConfigurationError(
            f"Unknown retriever implementation {retriever_type!r}"
        )

    def query_requirements(self, config: RetrieverConfig) -> frozenset[str]:
        """Return declarative requirements for a registered retriever definition."""
        implementation = config.implementation or config.type
        if implementation not in self.retrievers:
            raise ConfigurationError(
                f"Unknown retriever implementation {implementation!r}; available: "
                f"{', '.join(sorted(self.retrievers))}"
            )
        declared = frozenset(config.requirements.required)
        if declared == frozenset(
            {"dense_vector_index", "compatible_query_embedder"}
        ):
            defaults = {
                "postgres_all_chunks": frozenset({"persisted_chunks"}),
                "postgres_fts": frozenset({"persisted_chunks", "postgres_fts"}),
                "hybrid_rrf": frozenset(
                    {
                        "persisted_chunks",
                        "postgres_fts",
                        "dense_vector_index",
                        "compatible_query_embedder",
                    }
                ),
            }
            return defaults.get(implementation, declared)
        return declared

    def build_reranker(self, config: RerankerConfig) -> Reranker:
        """Construct one reranker from validated settings without loading models."""
        try:
            factory = self.rerankers[config.implementation or config.type]
        except KeyError as exc:
            raise ConfigurationError(
                f"Unknown reranker implementation {config.implementation!r}; "
                f"available: {', '.join(self.rerankers)}"
            ) from exc
        return factory(config)

    def build_knowledge_builder(
        self, component: ComponentConfig | str | None
    ) -> KnowledgeBuilder | None:
        """Construct the configured knowledge builder, or no stage when omitted."""
        if component is None:
            return None
        component_type = (
            component.implementation or component.type
            if isinstance(component, ComponentConfig)
            else component
        )
        try:
            factory = self.knowledge_builders[component_type]
        except KeyError as exc:
            raise ConfigurationError(
                f"Unknown knowledge_builder component {component_type!r}; "
                f"available: {', '.join(self.knowledge_builders)}"
            ) from exc
        return factory()
