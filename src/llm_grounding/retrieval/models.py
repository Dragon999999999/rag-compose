"""Canonical request, dependency-context, and response envelopes for retrieval."""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, overload

from llm_grounding.domain.corpus import (
    ChunkRetrievalResult,
    CorpusRequest,
    TaskType,
)
from llm_grounding.domain.evidence import RetrievalResult
from llm_grounding.embedding.models import EmbeddingVector

if TYPE_CHECKING:
    from llm_grounding.embedding.protocol import Embedder
    from llm_grounding.infrastructure.database.repositories import PostgresRepository
    from llm_grounding.infrastructure.vector.protocol import VectorStore
    from llm_grounding.ingestion.models import EmbeddingConfig


@dataclass(frozen=True)
class RetrievalRequest:
    """Select project-scoped evidence for one query.

    Retriever implementations receive this stable query envelope regardless of
    whether they use PostgreSQL, a vector backend, or a future hybrid index.
    ``top_k`` is optional so a retriever can apply its configured default.
    """

    project_id: str
    query: str
    document_ids: tuple[str, ...] = ()
    top_k: int | None = None
    task_type: TaskType = TaskType.QA

    def __post_init__(self) -> None:
        """Reject invalid project, query, filter, and result-limit values early."""
        if not self.project_id.strip():
            raise ValueError("project_id must be non-empty")
        if not self.query.strip():
            raise ValueError("query must be non-empty")
        if self.top_k is not None and self.top_k < 1:
            raise ValueError("top_k must be positive")

    @classmethod
    def from_corpus(
        cls, request: CorpusRequest, *, top_k: int | None = None
    ) -> "RetrievalRequest":
        """Adapt the existing grounding request into the canonical retriever form."""
        return cls(
            request.project_id,
            request.task,
            request.document_ids,
            top_k,
            request.task_type,
        )


@dataclass(frozen=True)
class RetrievalContext:
    """Provide resolved resources to a retriever without exposing construction logic."""

    repository: "PostgresRepository | None" = None
    vector_store: "VectorStore | None" = None
    embedder: "Embedder | None" = None
    embedding_config: "EmbeddingConfig | None" = None
    index_capabilities: object | None = None


@dataclass(frozen=True)
class RetrievalResponse:
    """Return ranked chunks with text, provenance, metadata, and optional score."""

    query: str
    results: tuple[ChunkRetrievalResult, ...]
    query_embedding: EmbeddingVector | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __iter__(self) -> Iterator[ChunkRetrievalResult]:
        """Allow legacy callers to iterate over the response's result collection."""
        return iter(self.results)

    def __len__(self) -> int:
        """Return the number of retrieved chunks for legacy sequence callers."""
        return len(self.results)

    @overload
    def __getitem__(self, index: int) -> ChunkRetrievalResult: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[ChunkRetrievalResult, ...]: ...

    def __getitem__(
        self, index: int | slice
    ) -> ChunkRetrievalResult | tuple[ChunkRetrievalResult, ...]:
        """Expose results by index while retaining the canonical response envelope."""
        return self.results[index]


@dataclass(frozen=True)
class RerankResponse:
    """Carry canonical evidence results through the existing reranker boundary."""

    query: str
    results: tuple[RetrievalResult, ...]
