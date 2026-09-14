"""Persistent corpus identities, derived chunks, and trusted query context.

Projects associate reusable documents with a selected ready ingestion. Chats own
conversation only. Retrieval units derive from canonical EvidenceUnits; their
source offsets are Unicode character offsets within the original evidence text.
No database, model, or vector-vendor objects cross this boundary.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum


@dataclass(frozen=True)
class Project:
    """Name a reusable document corpus independently of chats."""

    id: str
    name: str
    created_at: datetime


@dataclass(frozen=True)
class Document:
    """Identify immutable PDF content and its current user-facing filename."""

    id: str
    name: str
    sha256: str
    artifact_key: str
    created_at: datetime


@dataclass(frozen=True)
class ProjectDocument:
    """Associate a document with a project and its selected ready representation."""

    project_id: str
    document_id: str
    ingestion_id: str | None = None


class IngestionStatus(StrEnum):
    """Expose recoverable ingestion stages; only READY is queryable."""

    PENDING = "pending"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    READY = "ready"
    FAILED = "failed"


class SummaryStatus(StrEnum):
    """Expose the persisted lifecycle of a grounded document summary."""

    PENDING = "pending"
    GENERATING = "generating"
    READY = "ready"
    FAILED = "failed"


@dataclass(frozen=True)
class DocumentIngestion:
    """Identify a document representation and its effective configuration."""

    id: str
    document_id: str
    config_hash: str
    config_json: str
    embedding_hash: str
    dimension: int
    status: IngestionStatus
    created_at: datetime
    error: str | None = None


@dataclass(frozen=True)
class DocumentSummary:
    """Persist one grounded overview for a document ingestion representation."""

    document_id: str
    ingestion_id: str
    status: SummaryStatus
    rag_run_id: str | None
    error: str | None
    page_count: int | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class EvidenceReference:
    """Locate the portion of canonical evidence contributing to a chunk."""

    evidence_id: str
    start: int
    end: int


@dataclass(frozen=True)
class RetrievalUnit:
    """Derived retrieval context retaining evidence mappings and display provenance."""

    chunk_id: str
    document_id: str
    ingestion_id: str
    document_name: str
    text: str
    order: int
    page_start: int | None
    page_end: int | None
    evidence_refs: tuple[EvidenceReference, ...]
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ChunkRetrievalResult:
    """Attach query-specific cosine similarity and rank to a derived chunk."""

    chunk: RetrievalUnit
    rank: int
    retriever_id: str
    score: float | None = None


@dataclass(frozen=True)
class Chat:
    """Hold a project conversation and its automatic-name eligibility.

    Once `chat_name_set` is true, generated names must never replace the current
    name, even when its text happens to match the initial placeholder.
    """

    id: str
    project_id: str
    name: str
    created_at: datetime
    chat_name_set: bool = False


@dataclass(frozen=True)
class Message:
    """Persist conversational text; assistant messages are not trusted sources."""

    id: str
    chat_id: str
    role: str
    content: str
    created_at: datetime
    rag_run_id: str | None = None
    turn_title: str | None = None


class TaskType(StrEnum):
    """Distinguish question answering from summary retrieval planning."""

    QA = "qa"
    SUMMARY = "summary"


@dataclass(frozen=True)
class CorpusRequest:
    """Select a project and optional documents for a semantic task."""

    project_id: str
    task: str
    task_type: TaskType = TaskType.QA
    document_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class GroundingContext:
    """Supply selected chunks and separately labeled conversation to run stages.

    The runner verifies corpus membership before constructing this context.
    Assessors also consult canonical membership when validating nominated IDs.
    """

    request: CorpusRequest
    chunks: tuple[ChunkRetrievalResult, ...]
    history: tuple[Message, ...] = ()
    generate_chat_name: bool = False
