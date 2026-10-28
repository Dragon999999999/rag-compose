"""Canonical evidence and query-specific selections shared across stages.

Evidence remains immutable and independent of retrieval scores. RetrievalResult
is shared here because both completed runs and downstream stages retain it;
retrieval request/response envelopes remain local to retrieval.
"""

import math
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
class RankingScore:
    """Describe a native ranking score without changing its numeric semantics."""

    value: float
    kind: str
    method_id: str
    higher_is_better: bool = True
    calibrated: bool = False

    def __post_init__(self) -> None:
        """Reject non-finite values and empty score identities."""
        if not math.isfinite(self.value):
            raise ValueError("Ranking scores must be finite")
        if not self.kind.strip() or not self.method_id.strip():
            raise ValueError("Ranking score kind and method_id must be non-empty")


def score_kind(method_id: str) -> str:
    """Return the native score kind used by a registered retrieval method."""
    return {
        "vector": "cosine_similarity",
        "qdrant": "cosine_similarity",
        "pgvector": "cosine_similarity",
        "faiss": "cosine_similarity",
        "postgres_fts": "ts_rank_cd",
        "bm25": "bm25",
        "bge_m3_sparse": "learned_sparse_dot_product",
        "colbert_v2": "colbert_maxsim",
        "hybrid_rrf": "rrf",
        "hybrid": "rrf",
        "hybrid_weighted": "weighted_normalized_score",
    }.get(method_id, "retrieval")


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
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Detach metadata from callers so canonical evidence cannot be mutated."""
        import json

        metadata = dict(self.metadata)
        try:
            json.dumps(metadata)
        except TypeError as exc:
            raise ValueError("Evidence metadata must be JSON serializable") from exc
        object.__setattr__(self, "metadata", MappingProxyType(metadata))


@dataclass(frozen=True)
class RetrievalResult:
    """Record query-specific rank (one-based) and optional score for evidence."""

    evidence: EvidenceUnit
    rank: int
    retriever_id: str
    score: float | None = None
    query_variant: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    ranking_score: RankingScore | None = None
    rerank_score: RankingScore | None = None

    def __post_init__(self) -> None:
        """Derive a legacy retrieval-score record when a score is present."""
        if self.ranking_score is None and self.score is not None:
            object.__setattr__(
                self,
                "ranking_score",
                RankingScore(
                    self.score, score_kind(self.retriever_id), self.retriever_id
                ),
            )


def source_order(evidence: EvidenceUnit) -> tuple[str, int, int, str]:
    """Order documents by ID, then known pages and parser order, with stable ties."""
    return (
        evidence.document_id,
        evidence.page_number if evidence.page_number is not None else 0,
        evidence.order,
        evidence.evidence_id,
    )
