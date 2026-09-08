"""Query envelopes for selecting evidence; ranks live on shared RetrievalResult."""

from dataclasses import dataclass

from llm_grounding.domain.evidence import RetrievalResult
from llm_grounding.knowledge.models import KnowledgeBase


@dataclass(frozen=True)
class RetrievalRequest:
    """Select evidence for a semantic query from a supplied knowledge snapshot."""

    query: str
    knowledge: KnowledgeBase


@dataclass(frozen=True)
class RetrievalResponse:
    """Preserve the query and ordered candidates without modifying source evidence."""

    query: str
    results: tuple[RetrievalResult, ...]
