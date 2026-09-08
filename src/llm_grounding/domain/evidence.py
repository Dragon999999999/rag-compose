"""Canonical evidence and query-specific selections shared across stages.

Evidence remains immutable and independent of retrieval scores. RetrievalResult
is shared here because both completed runs and downstream stages retain it;
retrieval request/response envelopes remain local to retrieval.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType


class EvidenceType(StrEnum):
    """Identify the source modality without requiring text for every modality."""

    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"
    EQUATION = "equation"
    CAPTION = "caption"


@dataclass(frozen=True)
class EvidenceUnit:
    """Canonical source evidence; metadata is copied into a read-only mapping.

    Order is the parser's document-wide reading order. Source URI, page, block,
    and optional bounding box locate the evidence without claiming verification.
    """

    evidence_id: str
    document_id: str
    evidence_type: EvidenceType
    source_uri: str
    order: int
    page_number: int | None = None
    block_index: int | None = None
    bounding_box: tuple[float, float, float, float] | None = None
    text: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Detach metadata from callers so canonical evidence cannot be mutated."""
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class RetrievalResult:
    """Record query-specific rank (one-based) and optional score for evidence."""

    evidence: EvidenceUnit
    rank: int
    retriever_id: str
    score: float | None = None
    query_variant: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


def source_order(evidence: EvidenceUnit) -> tuple[str, int, int, str]:
    """Order documents by ID, then known pages and parser order, with stable ties."""
    return (
        evidence.document_id,
        evidence.page_number if evidence.page_number is not None else 0,
        evidence.order,
        evidence.evidence_id,
    )
