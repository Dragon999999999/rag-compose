"""Small typed factory registry for configured components, without dynamic plugins.

Validate every identifier before constructing anything. Alternative implementations
can be added to the corresponding typed mapping; composition never branches on
architecture names. A supplied model client replaces external generation for tests.
"""

from collections.abc import Callable, Collection
from dataclasses import dataclass
from typing import cast

from llm_grounding.assessment.deterministic import (
    DiversityAssessor,
    EvidenceSufficiencyAssessor,
    NoOpAssessor,
    RelevanceAssessor,
    RetrievalConfidenceAssessor,
)
from llm_grounding.assessment.protocol import Assessor
from llm_grounding.assessment.source_reference import SourceReferenceAssessor
from llm_grounding.decision.basic_grounding import BasicGroundingPolicy
from llm_grounding.embedding.factory import build_embedder
from llm_grounding.embedding.protocol import Embedder, SparseEmbedder, TokenEmbedder
from llm_grounding.infrastructure.config.loader import ConfigurationError
from llm_grounding.infrastructure.config.models import (
    AssessorConfig,
    ComponentConfig,
    KnowledgeBuilderConfig,
    ModelCatalog,
    PipelineConfig,
    ReasonerConfig,
    RerankerConfig,
    RetrieverConfig,
)
from llm_grounding.infrastructure.database.repositories import PostgresRepository
from llm_grounding.infrastructure.llm.aqueduct import AqueductModelClient
from llm_grounding.infrastructure.model_management.manager import LocalModelManager
from llm_grounding.infrastructure.prompts import PromptLibrary, PromptLibraryError
from llm_grounding.infrastructure.vector.protocol import VectorStore
from llm_grounding.ingestion.models import ChunkerConfig, EmbeddingConfig
from llm_grounding.knowledge.basic import BasicKnowledgeBuilder
from llm_grounding.knowledge.chunking.block_aware import BlockAwareChunker
from llm_grounding.knowledge.chunking.fixed_window import FixedWindowChunker
from llm_grounding.knowledge.chunking.hierarchical import HierarchicalChunker
from llm_grounding.knowledge.chunking.late_chunking import LateChunker
from llm_grounding.knowledge.chunking.llm_guided import (
    BoundaryGenerator,
    BoundaryOutput,
    LLMGuidedChunker,
    LocalBoundaryGenerator,
)
from llm_grounding.knowledge.chunking.protocol import Chunker
from llm_grounding.knowledge.chunking.recursive_token import RecursiveTokenChunker
from llm_grounding.knowledge.chunking.semantic import SemanticChunker
from llm_grounding.knowledge.chunking.sentence_window import SentenceWindowChunker
from llm_grounding.knowledge.chunking.structural import StructuralChunker
from llm_grounding.knowledge.metadata_gliner import GlinerMetadataKnowledgeBuilder
from llm_grounding.knowledge.metadata_rules import RuleMetadataKnowledgeBuilder
from llm_grounding.knowledge.metadata_spacy import SpacyMetadataKnowledgeBuilder
from llm_grounding.knowledge.protocol import KnowledgeBuilder
from llm_grounding.knowledge.structural import StructuralKnowledgeBuilder
from llm_grounding.parsing.protocol import DocumentParser
from llm_grounding.parsing.pymupdf import PyMuPDFDocumentParser
from llm_grounding.reasoning.basic_grounded import BasicGroundedReasoner
from llm_grounding.rendering.grounded_answer import GroundedAnswerRenderer
from llm_grounding.retrieval.bm25 import BM25Retriever
from llm_grounding.retrieval.colbert import ColBERTRetriever, MultiVectorIndex
from llm_grounding.retrieval.full_context import FullContextRetriever
from llm_grounding.retrieval.hybrid import HybridRetriever
from llm_grounding.retrieval.hybrid_weighted import WeightedHybridRetriever
from llm_grounding.retrieval.learned_sparse import (
    BgeM3SparseRetriever,
    SpladeV3Retriever,
)
from llm_grounding.retrieval.protocol import Retriever
from llm_grounding.retrieval.reranking.cross_encoder import CrossEncoderReranker
from llm_grounding.retrieval.reranking.identity import IdentityReranker
from llm_grounding.retrieval.reranking.local_model import LocalModelReranker
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
    sparse_embedder: SparseEmbedder | None = None
    sparse_store: object | None = None
    token_embedder: TokenEmbedder | None = None
    multivector_store: object | None = None


class _RemoteBoundaryGenerator:
    """Adapt the existing structured remote model client to chunking boundaries."""

    def __init__(self, client: AqueductModelClient) -> None:
        """Retain a credential-free remote client until chunking is requested."""
        self._client = client

    async def generate(self, prompt: str, max_new_tokens: int) -> str:
        """Request validated boundary JSON from the configured remote model."""
        result = await self._client.generate(prompt, BoundaryOutput)
        return result.output.model_dump_json()


def _vector_retriever_requires_runtime_dependencies() -> Retriever:
    """Reserve the registered vector stage for the Qdrant composition root."""
    raise ConfigurationError(
        "The vector retriever requires a configured vector backend and embedder; "
        "construct it through the application bootstrap."
    )


def _retriever_requires_runtime_dependencies(name: str) -> Retriever:
    """Reserve a retriever name for construction by the application composition root."""
    raise ConfigurationError(
        f"The {name} retriever requires project-scoped runtime dependencies; "
        "construct it through the application bootstrap."
    )


def _chunker_requires_runtime_dependencies(name: str) -> Chunker:
    """Reserve a model-backed chunker for the dependency-aware factory path."""
    raise ConfigurationError(
        f"The {name} chunker requires model-scoped runtime dependencies; "
        "construct it through the application bootstrap."
    )


class ComponentRegistry:
    """Map stage-local names to typed constructors and validate complete composition."""

    def __init__(self) -> None:
        """Register the supported split-pipeline implementations."""
        self.parsers: dict[str, Callable[[], DocumentParser]] = {
            "pymupdf": PyMuPDFDocumentParser
        }
        self.knowledge_builders: dict[
            str, Callable[[KnowledgeBuilderConfig], KnowledgeBuilder]
        ] = {
            "basic": lambda config: BasicKnowledgeBuilder(),
            "structural": lambda config: StructuralKnowledgeBuilder(),
            "metadata_rules": RuleMetadataKnowledgeBuilder,
            "metadata_spacy": SpacyMetadataKnowledgeBuilder,
        }
        self.retrievers: dict[str, Callable[[], Retriever]] = {
            "vector_similarity": _vector_retriever_requires_runtime_dependencies,
            "postgres_all_chunks": lambda: _retriever_requires_runtime_dependencies(
                "postgres_all_chunks"
            ),
            "postgres_fts": lambda: _retriever_requires_runtime_dependencies(
                "postgres_fts"
            ),
            "bm25": lambda: _retriever_requires_runtime_dependencies("bm25"),
            "splade_v3": lambda: _retriever_requires_runtime_dependencies("splade_v3"),
            "bge_m3_sparse": lambda: _retriever_requires_runtime_dependencies(
                "bge_m3_sparse"
            ),
            "colbert_v2": lambda: _retriever_requires_runtime_dependencies(
                "colbert_v2"
            ),
            "hybrid_rrf": lambda: _retriever_requires_runtime_dependencies(
                "hybrid_rrf"
            ),
            "hybrid_weighted": lambda: _retriever_requires_runtime_dependencies(
                "hybrid_weighted"
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
        self.pipeline_assessors = frozenset(
            {
                "noop",
                "source_reference",
                "retrieval_confidence",
                "relevance",
                "evidence_sufficiency",
                "diversity",
            }
        )
        self.pipeline_decision_policies = frozenset({"basic_grounding"})
        self.pipeline_renderers = frozenset({"grounded_answer"})
        self.chunkers: dict[str, Callable[[ChunkerConfig], Chunker]] = {
            "block_aware": BlockAwareChunker,
            "fixed_window": FixedWindowChunker,
            "recursive_token": RecursiveTokenChunker,
            "hierarchical": HierarchicalChunker,
            "structural": StructuralChunker,
            "sentence_window": SentenceWindowChunker,
            "semantic": lambda config: _chunker_requires_runtime_dependencies(
                "semantic"
            ),
            "llm_guided": lambda config: _chunker_requires_runtime_dependencies(
                "llm_guided"
            ),
            "late_chunking": lambda config: _chunker_requires_runtime_dependencies(
                "late_chunking"
            ),
        }
        self._retriever_factories: dict[
            str, Callable[[RetrieverConfig, RetrieverDependencies], Retriever]
        ] = {
            "vector_similarity": self._build_vector_retriever,
            "postgres_all_chunks": self._build_full_context_retriever,
            "postgres_fts": self._build_sparse_retriever,
            "bm25": self._build_bm25_retriever,
            "splade_v3": self._build_splade_retriever,
            "bge_m3_sparse": self._build_bge_sparse_retriever,
            "colbert_v2": self._build_colbert_retriever,
            "hybrid_rrf": self._build_hybrid_retriever,
            "hybrid_weighted": self._build_weighted_hybrid_retriever,
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

    def build_chunker(
        self,
        config: ChunkerConfig,
        *,
        model_catalog: ModelCatalog | None = None,
        local_model_manager: LocalModelManager | None = None,
    ) -> Chunker:
        """Construct a chunker and resolve any model through shared providers."""
        implementation = config.implementation or config.type
        if implementation in {"semantic", "late_chunking"}:
            if model_catalog is None or local_model_manager is None:
                raise ConfigurationError(
                    f"{implementation} chunking requires model catalog and manager"
                )
            if config.model_id is None:
                raise ConfigurationError(f"{implementation} chunking requires model_id")
            embedding_config = model_catalog.embedding(config.model_id)
            if implementation == "semantic":
                embedder = build_embedder(embedding_config, local_model_manager)
                if not hasattr(embedder, "embed_documents"):
                    raise ConfigurationError(
                        "Semantic chunking requires a dense embedding provider"
                    )
                return SemanticChunker(config, cast(Embedder, embedder))
            token_config = embedding_config.model_copy(
                update={"representation": "multivector"}
            )
            if not embedding_config.late_chunking:
                raise ConfigurationError(
                    f"Embedding model {config.model_id} does not declare "
                    "late_chunking support"
                )
            token_embedder = build_embedder(token_config, local_model_manager)
            if not hasattr(token_embedder, "embed_token_documents"):
                raise ConfigurationError(
                    "Late chunking requires a token-capable embedding provider"
                )
            return LateChunker(
                config,
                cast(TokenEmbedder, token_embedder),
                embedding_config.vector_identity(),
            )
        if implementation == "llm_guided":
            if config.model_id is None:
                raise ConfigurationError("llm_guided chunking requires model_id")
            generator: BoundaryGenerator
            if (
                model_catalog is not None
                and config.model_id in model_catalog.reasoning_models
            ):
                generator = _RemoteBoundaryGenerator(
                    AqueductModelClient(model_catalog.reasoning(config.model_id))
                )
            elif local_model_manager is not None:
                generator = LocalBoundaryGenerator(config.model_id, local_model_manager)
            else:
                raise ConfigurationError(
                    "llm_guided chunking requires a remote model or LocalModelManager"
                )
            return LLMGuidedChunker(config, generator)
        try:
            factory = self.chunkers[implementation]
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
            raise ConfigurationError(f"Unknown reasoner component {config.type!r}")
        if config.prompt_id is None:
            raise ConfigurationError("The basic_grounded reasoner requires prompt_id")
        return BasicGroundedReasoner(
            AqueductModelClient(config.model),
            PromptLibrary.default(),
            config.prompt_id,
        )

    def build_assessor(
        self,
        config: AssessorConfig | ComponentConfig,
        repository: PostgresRepository | None = None,
    ) -> Assessor:
        """Construct one registered deterministic evidence assessor."""
        implementation = config.implementation or config.type
        if implementation == "noop":
            return NoOpAssessor()
        if implementation == "source_reference":
            return SourceReferenceAssessor(repository)
        if not isinstance(config, AssessorConfig):
            raise ConfigurationError(
                f"Assessor {implementation!r} requires typed configuration"
            )
        if implementation == "retrieval_confidence":
            return RetrievalConfidenceAssessor(
                threshold=config.threshold,
                score_kind=config.score_kind,
                min_threshold_count=config.min_threshold_count,
            )
        if implementation == "relevance":
            return RelevanceAssessor(
                threshold=config.threshold,
                fallback_to_retrieval=config.fallback_to_retrieval,
            )
        if implementation == "evidence_sufficiency":
            return EvidenceSufficiencyAssessor(
                requested_aspects=config.requested_aspects,
                threshold=config.threshold if config.threshold is not None else 0.5,
            )
        if implementation == "diversity":
            return DiversityAssessor(max_pairs=config.max_pairs)
        raise ConfigurationError(f"Unknown assessor component {implementation!r}")

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

    @staticmethod
    def _build_bm25_retriever(
        config: RetrieverConfig, dependencies: RetrieverDependencies
    ) -> Retriever:
        """Build genuine BM25 retrieval over canonical chunk text."""
        return BM25Retriever(
            dependencies.repository,
            config.top_k or 20,
            k1=config.bm25_k1,
            b=config.bm25_b,
        )

    @staticmethod
    def _build_splade_retriever(
        config: RetrieverConfig, dependencies: RetrieverDependencies
    ) -> Retriever:
        """Build SPLADE-v3 retrieval from a lazily provisioned sparse index."""
        if dependencies.sparse_embedder is None or dependencies.sparse_store is None:
            raise ConfigurationError(
                "SPLADE-v3 retrieval requires a sparse model and index"
            )
        return SpladeV3Retriever(
            dependencies.repository,
            dependencies.sparse_embedder,
            dependencies.sparse_store,
            top_k=config.top_k or 20,
        )

    @staticmethod
    def _build_bge_sparse_retriever(
        config: RetrieverConfig, dependencies: RetrieverDependencies
    ) -> Retriever:
        """Build BGE-M3 sparse retrieval using the shared BGE model lease."""
        if dependencies.sparse_embedder is None or dependencies.sparse_store is None:
            raise ConfigurationError(
                "BGE-M3 sparse retrieval requires a sparse model and index"
            )
        return BgeM3SparseRetriever(
            dependencies.repository,
            dependencies.sparse_embedder,
            dependencies.sparse_store,
            top_k=config.top_k or 20,
        )

    @staticmethod
    def _build_colbert_retriever(
        config: RetrieverConfig, dependencies: RetrieverDependencies
    ) -> Retriever:
        """Build ColBERT-v2 retrieval backed by a Qdrant MaxSim index."""
        if (
            dependencies.token_embedder is None
            or dependencies.multivector_store is None
        ):
            raise ConfigurationError(
                "ColBERT-v2 retrieval requires a token model and index"
            )
        return ColBERTRetriever(
            dependencies.repository,
            dependencies.token_embedder,
            cast(MultiVectorIndex, dependencies.multivector_store),
            candidate_top_k=config.candidate_top_k or config.sparse_top_k or 50,
            top_k=config.top_k or 10,
        )

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

    def _build_weighted_hybrid_retriever(
        self, config: RetrieverConfig, dependencies: RetrieverDependencies
    ) -> Retriever:
        """Compose dense retrieval with BM25 using normalized score fusion."""
        dense = self._build_vector_retriever(
            RetrieverConfig(
                type="vector",
                top_k=config.dense_top_k or 20,
                requirements=config.requirements,
            ),
            dependencies,
        )
        sparse = self._build_bm25_retriever(config, dependencies)
        return WeightedHybridRetriever(
            dense,
            sparse,
            dense_top_k=config.dense_top_k or 20,
            sparse_top_k=config.sparse_top_k or 20,
            top_k=config.top_k or 10,
            dense_weight=config.dense_weight,
            sparse_weight=config.sparse_weight,
            score_normalization=config.score_normalization,
        )

    def query_capabilities(self, retriever_type: str) -> frozenset[str]:
        """Return resource capabilities for a registered retriever implementation."""
        if retriever_type == "vector_similarity":
            return frozenset({"chunk_vectors", "dense_vector_index"})
        if retriever_type == "postgres_all_chunks":
            return frozenset({"persisted_chunks"})
        if retriever_type == "postgres_fts":
            return frozenset({"persisted_chunks", "postgres_fts"})
        if retriever_type == "bm25":
            return frozenset({"persisted_chunks"})
        if retriever_type in {"splade_v3", "bge_m3_sparse"}:
            return frozenset(
                {"persisted_chunks", "sparse_vector_index", "sparse_query_embedder"}
            )
        if retriever_type == "colbert_v2":
            return frozenset(
                {"persisted_chunks", "multivector_index", "late_interaction_model"}
            )
        if retriever_type == "hybrid_rrf":
            return frozenset(
                {
                    "persisted_chunks",
                    "postgres_fts",
                    "chunk_vectors",
                    "dense_vector_index",
                }
            )
        if retriever_type == "hybrid_weighted":
            return frozenset(
                {
                    "persisted_chunks",
                    "chunk_vectors",
                    "dense_vector_index",
                    "compatible_query_embedder",
                }
            )
        raise ConfigurationError(f"Unknown retriever implementation {retriever_type!r}")

    def query_requirements(self, config: RetrieverConfig) -> frozenset[str]:
        """Return declarative requirements for a registered retriever definition."""
        implementation = config.implementation or config.type
        if implementation not in self.retrievers:
            raise ConfigurationError(
                f"Unknown retriever implementation {implementation!r}; available: "
                f"{', '.join(sorted(self.retrievers))}"
            )
        declared = frozenset(config.requirements.required)
        if declared == frozenset({"dense_vector_index", "compatible_query_embedder"}):
            defaults = {
                "postgres_all_chunks": frozenset({"persisted_chunks"}),
                "postgres_fts": frozenset({"persisted_chunks", "postgres_fts"}),
                "bm25": frozenset({"persisted_chunks"}),
                "splade_v3": frozenset(
                    {"persisted_chunks", "sparse_vector_index", "sparse_query_embedder"}
                ),
                "bge_m3_sparse": frozenset(
                    {"persisted_chunks", "sparse_vector_index", "sparse_query_embedder"}
                ),
                "colbert_v2": frozenset(
                    {"persisted_chunks", "multivector_index", "late_interaction_model"}
                ),
                "hybrid_rrf": frozenset(
                    {
                        "persisted_chunks",
                        "postgres_fts",
                        "dense_vector_index",
                        "compatible_query_embedder",
                    }
                ),
                "hybrid_weighted": frozenset(
                    {
                        "persisted_chunks",
                        "dense_vector_index",
                        "compatible_query_embedder",
                    }
                ),
            }
            return defaults.get(implementation, declared)
        return declared

    def build_reranker(
        self,
        config: RerankerConfig,
        local_model_manager: "LocalModelManager | None" = None,
    ) -> Reranker:
        """Construct one reranker from validated settings without loading models."""
        if config.implementation in {"cross_encoder", "generative"} and config.model_id:
            if local_model_manager is None:
                raise ConfigurationError(
                    f"Reranker {config.model_id!r} requires LocalModelManager"
                )
            return LocalModelReranker(
                config.model_id,
                local_model_manager,
                top_k=config.rerank_top_k,
            )
        try:
            factory = self.rerankers[config.implementation or config.type]
        except KeyError as exc:
            raise ConfigurationError(
                f"Unknown reranker implementation {config.implementation!r}; "
                f"available: {', '.join(self.rerankers)}"
            ) from exc
        return factory(config)

    def build_knowledge_builder(
        self,
        component: KnowledgeBuilderConfig | ComponentConfig | str | None,
        *,
        local_model_manager: "LocalModelManager | None" = None,
    ) -> KnowledgeBuilder | None:
        """Construct a builder while keeping model lifecycle in LocalModelManager."""
        if component is None:
            return None
        if isinstance(component, str):
            config = KnowledgeBuilderConfig.model_validate(
                {"implementation": component}
            )
        elif isinstance(component, KnowledgeBuilderConfig):
            config = component
        else:
            config = KnowledgeBuilderConfig.model_validate(component.model_dump())
        component_type = config.implementation
        if component_type == "metadata_gliner":
            if local_model_manager is None:
                raise ConfigurationError(
                    "GLiNER metadata enrichment requires LocalModelManager"
                )
            return GlinerMetadataKnowledgeBuilder(config, local_model_manager)
        try:
            factory = self.knowledge_builders[component_type]
        except KeyError as exc:
            raise ConfigurationError(
                f"Unknown knowledge_builder component {component_type!r}; "
                f"available: {', '.join(self.knowledge_builders)}"
            ) from exc
        return factory(config)
