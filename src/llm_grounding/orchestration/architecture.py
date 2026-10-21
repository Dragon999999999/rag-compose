"""Typed component composition; implementations do not see the full architecture."""

from dataclasses import dataclass

from llm_grounding.assessment.protocol import Assessor
from llm_grounding.decision.protocol import DecisionPolicy
from llm_grounding.embedding.protocol import Embedder
from llm_grounding.knowledge.protocol import KnowledgeBuilder
from llm_grounding.parsing.protocol import DocumentParser
from llm_grounding.reasoning.protocol import ReasoningEngine
from llm_grounding.rendering.protocol import Renderer
from llm_grounding.retrieval.protocol import Retriever
from llm_grounding.retrieval.reranking.protocol import Reranker


@dataclass(frozen=True)
class Architecture:
    """Supply interchangeable stages to an orchestration strategy."""

    name: str
    version: int
    parser: DocumentParser
    knowledge_builder: KnowledgeBuilder
    retriever: Retriever
    reranker: Reranker
    reasoner: ReasoningEngine
    assessors: tuple[Assessor, ...]
    decision_policy: DecisionPolicy
    renderer: Renderer
    embedder: Embedder | None = None
