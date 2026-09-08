"""In-memory knowledge representation passed from knowledge builders to retrieval."""

from dataclasses import dataclass

from llm_grounding.domain.evidence import EvidenceUnit


@dataclass(frozen=True)
class KnowledgeBase:
    """Hold canonical evidence independently of any query or storage technology."""

    evidence: tuple[EvidenceUnit, ...]
