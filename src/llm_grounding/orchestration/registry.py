"""Small typed factory registry for configured components, without dynamic plugins.

Validate every identifier before constructing anything. Alternative implementations
can be added to the corresponding typed mapping; composition never branches on
architecture names. A supplied model client replaces external generation for tests.
"""

from collections.abc import Callable, Collection

from llm_grounding.infrastructure.config.loader import ConfigurationError
from llm_grounding.infrastructure.config.models import (
    ModelCatalog,
    PipelineConfig,
    RerankerConfig,
    RetrieverConfig,
)
from llm_grounding.infrastructure.prompts import PromptLibrary, PromptLibraryError
from llm_grounding.ingestion.models import ChunkerConfig
from llm_grounding.knowledge.basic import BasicKnowledgeBuilder
from llm_grounding.knowledge.chunking.block_aware import BlockAwareChunker
from llm_grounding.knowledge.chunking.fixed_window import FixedWindowChunker
from llm_grounding.knowledge.chunking.protocol import Chunker
from llm_grounding.knowledge.protocol import KnowledgeBuilder
from llm_grounding.parsing.protocol import DocumentParser
from llm_grounding.parsing.pymupdf import PyMuPDFDocumentParser
from llm_grounding.retrieval.protocol import Retriever
from llm_grounding.retrieval.reranking.cross_encoder import CrossEncoderReranker
from llm_grounding.retrieval.reranking.identity import IdentityReranker
from llm_grounding.retrieval.reranking.protocol import Reranker
from llm_grounding.retrieval.reranking.top_k import TopKReranker


def _vector_retriever_requires_runtime_dependencies() -> Retriever:
    """Reserve the registered vector stage for the Qdrant composition root."""
    raise ConfigurationError(
        "The vector retriever requires the Qdrant repository and embedder; "
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
            "vector": _vector_retriever_requires_runtime_dependencies,
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
        self.vector_backends: frozenset[str] = frozenset({"qdrant"})

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
                ("parser", config.representation.parser.type, self.parsers)
            )
            if config.representation.knowledge_builder is not None:
                selections.append(
                    (
                        "knowledge_builder",
                        config.representation.knowledge_builder.type,
                        self.knowledge_builders,
                    )
                )
        if config.query is not None:
            selections.extend(
                (
                    ("retriever", config.query.retriever.type, self.retrievers),
                    ("reranker", config.query.reranker.type, self.rerankers),
                    ("reasoner", config.query.reasoner.type, self.pipeline_reasoners),
                    (
                        "decision_policy",
                        config.query.decision_policy.type,
                        self.pipeline_decision_policies,
                    ),
                    ("renderer", config.query.renderer.type, self.pipeline_renderers),
                )
            )
            selections.extend(
                ("assessor", item.type, self.pipeline_assessors)
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
            and config.representation.chunker.type not in self.chunkers
        ):
            raise ConfigurationError(
                f"Unknown chunker component {config.representation.chunker.type!r}; "
                f"available: {', '.join(self.chunkers)}"
            )
        if (
            config.representation is not None
            and config.representation.vector_backend is not None
            and config.representation.vector_backend.type not in self.vector_backends
        ):
            raise ConfigurationError(
                "Unknown vector_backend component "
                f"{config.representation.vector_backend.type!r}; available: "
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
            and config.query.retriever.type == "vector"
            and (
                config.representation is not None
                and config.representation.vector_backend is not None
                and config.representation.vector_backend.type != "qdrant"
            )
        ):
            raise ConfigurationError(
                "The vector retriever requires the qdrant vector backend"
            )
        if config.query is not None and (
            config.query.reasoner.type == "basic_grounded"
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

    def query_capabilities(self, retriever_type: str) -> frozenset[str]:
        """Return backward-compatible resource capabilities for a retriever type."""
        if retriever_type == "vector":
            return frozenset({"chunk_vectors", "dense_vector_index"})
        raise ConfigurationError(f"Unknown retriever component {retriever_type!r}")

    def query_requirements(self, config: RetrieverConfig) -> frozenset[str]:
        """Return declarative requirements for a registered retriever definition."""
        if config.type not in self.retrievers:
            raise ConfigurationError(
                f"Unknown retriever component {config.type!r}; available: "
                f"{', '.join(sorted(self.retrievers))}"
            )
        return frozenset(config.requirements.required)

    def build_reranker(self, config: RerankerConfig) -> Reranker:
        """Construct one reranker from validated settings without loading models."""
        try:
            factory = self.rerankers[config.type]
        except KeyError as exc:
            raise ConfigurationError(
                f"Unknown reranker component {config.type!r}; "
                f"available: {', '.join(self.rerankers)}"
            ) from exc
        return factory(config)

    def build_knowledge_builder(
        self, component_type: str | None
    ) -> KnowledgeBuilder | None:
        """Construct the configured knowledge builder, or no stage when omitted."""
        if component_type is None:
            return None
        try:
            factory = self.knowledge_builders[component_type]
        except KeyError as exc:
            raise ConfigurationError(
                f"Unknown knowledge_builder component {component_type!r}; "
                f"available: {', '.join(self.knowledge_builders)}"
            ) from exc
        return factory()
