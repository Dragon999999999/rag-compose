"""Small typed factory registry for configured components, without dynamic plugins.

Validate every identifier before constructing anything. Alternative implementations
can be added to the corresponding typed mapping; composition never branches on
architecture names. A supplied model client replaces external generation for tests.
"""

from collections.abc import Callable, Collection

from llm_grounding.assessment.noop import NoOpAssessor
from llm_grounding.assessment.protocol import Assessor
from llm_grounding.decision.accept_all import AcceptAllDecisionPolicy
from llm_grounding.decision.protocol import DecisionPolicy
from llm_grounding.infrastructure.config.loader import ConfigurationError
from llm_grounding.infrastructure.config.models import ArchitectureConfig
from llm_grounding.infrastructure.llm.aqueduct import AqueductModelClient
from llm_grounding.infrastructure.llm.protocol import StructuredModelClient
from llm_grounding.knowledge.basic import BasicKnowledgeBuilder
from llm_grounding.knowledge.protocol import KnowledgeBuilder
from llm_grounding.orchestration.architecture import Architecture
from llm_grounding.parsing.protocol import DocumentParser
from llm_grounding.parsing.pymupdf import PyMuPDFDocumentParser
from llm_grounding.reasoning.basic_summary import BasicSummaryReasoner
from llm_grounding.reasoning.protocol import ReasoningEngine
from llm_grounding.rendering.basic_summary import BasicSummaryRenderer
from llm_grounding.rendering.protocol import Renderer
from llm_grounding.retrieval.full_document import FullDocumentRetriever
from llm_grounding.retrieval.protocol import Retriever
from llm_grounding.retrieval.reranking.identity import IdentityReranker
from llm_grounding.retrieval.reranking.protocol import Reranker


class ComponentRegistry:
    """Map stage-local names to typed constructors and validate complete composition."""

    def __init__(self) -> None:
        """Register the minimal baseline implementations without creating providers."""
        self.parsers: dict[str, Callable[[], DocumentParser]] = {
            "pymupdf": PyMuPDFDocumentParser
        }
        self.knowledge_builders: dict[str, Callable[[], KnowledgeBuilder]] = {
            "basic": BasicKnowledgeBuilder
        }
        self.retrievers: dict[str, Callable[[], Retriever]] = {
            "full_document": FullDocumentRetriever
        }
        self.rerankers: dict[str, Callable[[], Reranker]] = {
            "identity": IdentityReranker
        }
        self.reasoners: dict[
            str, Callable[[StructuredModelClient], ReasoningEngine]
        ] = {"basic_summary": BasicSummaryReasoner}
        self.assessors: dict[str, Callable[[], Assessor]] = {"noop": NoOpAssessor}
        self.decision_policies: dict[str, Callable[[], DecisionPolicy]] = {
            "accept_all": AcceptAllDecisionPolicy
        }
        self.renderers: dict[str, Callable[[], Renderer]] = {
            "basic_summary": BasicSummaryRenderer
        }

    def validate(self, config: ArchitectureConfig) -> None:
        """Reject unknown identifiers with stage context before constructing clients."""
        selections: list[tuple[str, str, Collection[str]]] = [
            ("parser", config.parser.type, self.parsers),
            (
                "knowledge_builder",
                config.knowledge_builder.type,
                self.knowledge_builders,
            ),
            ("retriever", config.retriever.type, self.retrievers),
            ("reranker", config.reranker.type, self.rerankers),
            ("reasoner", config.reasoner.type, self.reasoners),
            ("decision_policy", config.decision_policy.type, self.decision_policies),
            ("renderer", config.renderer.type, self.renderers),
        ]
        selections.extend(
            ("assessor", item.type, self.assessors) for item in config.assessors
        )
        for stage, name, factories in selections:
            if name not in factories:
                raise ConfigurationError(
                    f"Unknown {stage} component {name!r}; "
                    f"available: {', '.join(factories)}"
                )

    def build(
        self,
        config: ArchitectureConfig,
        model_client: StructuredModelClient | None = None,
    ) -> Architecture:
        """Resolve components and defer credential loading until generation."""
        self.validate(config)
        client = (
            model_client
            if model_client is not None
            else AqueductModelClient(config.reasoner.model)
        )
        return Architecture(
            name=config.name,
            version=config.version,
            parser=self.parsers[config.parser.type](),
            knowledge_builder=self.knowledge_builders[config.knowledge_builder.type](),
            retriever=self.retrievers[config.retriever.type](),
            reranker=self.rerankers[config.reranker.type](),
            reasoner=self.reasoners[config.reasoner.type](client),
            assessors=tuple(self.assessors[item.type]() for item in config.assessors),
            decision_policy=self.decision_policies[config.decision_policy.type](),
            renderer=self.renderers[config.renderer.type](),
        )
