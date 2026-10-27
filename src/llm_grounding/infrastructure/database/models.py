"""SQLAlchemy persistence models; domain objects are mapped at repository boundaries.

PostgreSQL is canonical for projects, document metadata, evidence/chunks, chats,
messages, and run metadata. Original PDFs remain in artifact storage; Qdrant holds
only rebuildable vector-index entries. The schema intentionally omits advanced claims.
"""

from datetime import datetime
from uuid import uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    false,
    func,
    true,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _identifier() -> str:
    """Create a database-independent UUID string primary key."""
    return str(uuid4())


class Base(DeclarativeBase):
    """Base for canonical relational persistence tables."""


class ProjectRow(Base):
    """Persist a reusable corpus identity."""

    __tablename__ = "projects"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_identifier)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    ingestion_pipeline_id: Mapped[str | None] = mapped_column(
        ForeignKey("pipelines.id", ondelete="SET NULL")
    )
    ingestion_pipeline_version: Mapped[int | None] = mapped_column(Integer)
    ingestion_pipeline_config_hash: Mapped[str | None] = mapped_column(String(64))
    ingestion_representation_hash: Mapped[str | None] = mapped_column(String(64))
    index_capabilities_json: Mapped[dict[str, object]] = mapped_column(
        JSONB, nullable=False, default=dict
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class PipelineRow(Base):
    """Persist the PostgreSQL source of truth for named RAG pipelines."""

    __tablename__ = "pipelines"
    __table_args__ = (UniqueConstraint("name", name="uq_pipeline_name"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_identifier)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    schema_version: Mapped[int] = mapped_column(Integer, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    config_json: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    representation_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    source_kind: Mapped[str] = mapped_column(String(16), nullable=False)
    source_path: Mapped[str | None] = mapped_column(Text)
    source_fingerprint: Mapped[str | None] = mapped_column(String(64))
    active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default=true()
    )
    is_default: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=false()
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class PipelineVersionRow(Base):
    """Persist immutable pipeline configuration snapshots for reproducibility."""

    __tablename__ = "pipeline_versions"
    __table_args__ = (
        UniqueConstraint(
            "pipeline_id", "version", name="uq_pipeline_version_identity"
        ),
    )

    pipeline_id: Mapped[str] = mapped_column(
        ForeignKey("pipelines.id", ondelete="CASCADE"), primary_key=True
    )
    version: Mapped[int] = mapped_column(Integer, primary_key=True)
    schema_version: Mapped[int] = mapped_column(Integer, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    config_json: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    representation_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class DocumentRow(Base):
    """Persist deduplicated PDF identity, display name, and artifact reference."""

    __tablename__ = "documents"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_identifier)
    name: Mapped[str] = mapped_column(String(1024), nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    artifact_key: Mapped[str] = mapped_column(String(1024), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class ProjectDocumentRow(Base):
    """Allow one stored PDF to participate in multiple project corpora."""

    __tablename__ = "project_documents"
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True
    )
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True
    )
    ingestion_id: Mapped[str | None] = mapped_column(
        ForeignKey("document_ingestions.id", ondelete="SET NULL")
    )


class IngestionRow(Base):
    """Persist one immutable parsed/chunked/embedded document representation."""

    __tablename__ = "document_ingestions"
    __table_args__ = (
        UniqueConstraint(
            "document_id", "config_hash", name="uq_document_ingestion_config"
        ),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_identifier)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    config_json: Mapped[str] = mapped_column(Text, nullable=False)
    embedding_hash: Mapped[str | None] = mapped_column(String(64))
    dimension: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )


class IngestionObservationRow(Base):
    """Persist the derived evidence/chunk inventory for an ingestion."""

    __tablename__ = "ingestion_observations"
    ingestion_id: Mapped[str] = mapped_column(
        ForeignKey("document_ingestions.id", ondelete="CASCADE"), primary_key=True
    )
    observation_json: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class EvaluationCorpusRow(Base):
    """Persist isolated benchmark corpus inventories outside production projects."""

    __tablename__ = "evaluation_corpora"
    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    parent_corpus_id: Mapped[str | None] = mapped_column(String(255))
    corpus_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    documents_json: Mapped[list[dict[str, object]]] = mapped_column(JSON, default=list)
    chunks_json: Mapped[list[dict[str, object]]] = mapped_column(JSON, default=list)
    perturbation_metadata: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class EvaluationCorpusDocumentRow(Base):
    """Store isolated evaluation source-document metadata in its own namespace."""

    __tablename__ = "evaluation_corpus_documents"

    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    corpus_id: Mapped[str] = mapped_column(
        ForeignKey("evaluation_corpora.id", ondelete="CASCADE"), index=True
    )
    document_json: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False)


class EvaluationCorpusChunkRow(Base):
    """Store canonical benchmark chunks independently from production chunks."""

    __tablename__ = "evaluation_corpus_chunks"

    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    corpus_id: Mapped[str] = mapped_column(
        ForeignKey("evaluation_corpora.id", ondelete="CASCADE"), index=True
    )
    chunk_json: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False)


class EvaluationCorpusOperationRow(Base):
    """Track controlled corpus uploads and deletion operations."""

    __tablename__ = "evaluation_corpus_operations"

    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    corpus_id: Mapped[str] = mapped_column(String(255), index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    operation_json: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class EvidenceRow(Base):
    """Persist canonical text evidence and source provenance for an ingestion."""

    __tablename__ = "evidence_units"
    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    ingestion_id: Mapped[str] = mapped_column(
        ForeignKey("document_ingestions.id", ondelete="CASCADE"), nullable=False
    )
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    evidence_type: Mapped[str] = mapped_column(String(16), nullable=False)
    source_uri: Mapped[str] = mapped_column(Text, nullable=False)
    ordering: Mapped[int] = mapped_column(Integer, nullable=False)
    page_number: Mapped[int | None] = mapped_column(Integer)
    block_index: Mapped[int | None] = mapped_column(Integer)
    bounding_box: Mapped[list[float] | None] = mapped_column(JSON)
    text: Mapped[str | None] = mapped_column(Text)
    metadata_json: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)


class RetrievalUnitRow(Base):
    """Persist derived chunk text and display provenance; vectors remain derived."""

    __tablename__ = "retrieval_units"
    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    ingestion_id: Mapped[str] = mapped_column(
        ForeignKey("document_ingestions.id", ondelete="CASCADE"), nullable=False
    )
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    document_name: Mapped[str] = mapped_column(String(1024), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    ordering: Mapped[int] = mapped_column(Integer, nullable=False)
    page_start: Mapped[int | None] = mapped_column(Integer)
    page_end: Mapped[int | None] = mapped_column(Integer)
    metadata_json: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)


class RetrievalUnitMemberRow(Base):
    """Map chunk spans back to canonical evidence identifiers."""

    __tablename__ = "retrieval_unit_members"
    chunk_id: Mapped[str] = mapped_column(
        ForeignKey("retrieval_units.id", ondelete="CASCADE"), primary_key=True
    )
    evidence_id: Mapped[str] = mapped_column(
        ForeignKey("evidence_units.id", ondelete="CASCADE"), primary_key=True
    )
    start: Mapped[int] = mapped_column(Integer, primary_key=True)
    end: Mapped[int] = mapped_column(Integer, nullable=False)


class ChatRow(Base):
    """Persist a conversation, corpus scope, and selected pipeline snapshot."""

    __tablename__ = "chats"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_identifier)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    chat_name_set: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=false()
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    pipeline_id: Mapped[str | None] = mapped_column(
        ForeignKey("pipelines.id", ondelete="SET NULL")
    )
    pipeline_version: Mapped[int | None] = mapped_column(Integer)
    pipeline_config_hash: Mapped[str | None] = mapped_column(String(64))
    pipeline_representation_hash: Mapped[str | None] = mapped_column(String(64))


class RagRunRow(Base):
    """Persist run metadata while the detailed trace remains application-owned."""

    __tablename__ = "rag_runs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    project_id: Mapped[str | None] = mapped_column(
        ForeignKey("projects.id", ondelete="SET NULL")
    )
    architecture_name: Mapped[str] = mapped_column(String(255), nullable=False)
    architecture_version: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    pipeline_id: Mapped[str | None] = mapped_column(
        ForeignKey("pipelines.id", ondelete="SET NULL")
    )
    pipeline_version: Mapped[int | None] = mapped_column(Integer)
    pipeline_config_hash: Mapped[str | None] = mapped_column(String(64))
    pipeline_representation_hash: Mapped[str | None] = mapped_column(String(64))
    metadata_json: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class RagRunObservationRow(Base):
    """Persist the canonical evaluation observation for replay and re-scoring."""

    __tablename__ = "rag_run_observations"
    run_id: Mapped[str] = mapped_column(
        ForeignKey("rag_runs.id", ondelete="CASCADE"), primary_key=True
    )
    request_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    benchmark_case_id: Mapped[str | None] = mapped_column(String(255), index=True)
    experiment_id: Mapped[str | None] = mapped_column(String(255), index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    observation_json: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class DocumentSummaryRow(Base):
    """Persist a grounded document overview and its validated RAG run."""

    __tablename__ = "document_summaries"
    __table_args__ = (
        UniqueConstraint(
            "document_id", "ingestion_id", name="uq_document_summary_ingestion"
        ),
    )
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True
    )
    ingestion_id: Mapped[str] = mapped_column(
        ForeignKey("document_ingestions.id", ondelete="CASCADE"), primary_key=True
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    rag_run_id: Mapped[str | None] = mapped_column(
        ForeignKey("rag_runs.id", ondelete="SET NULL")
    )
    error: Mapped[str | None] = mapped_column(Text)
    page_count: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )


class MessageRow(Base):
    """Persist chat messages and optional link to their RAG run."""

    __tablename__ = "messages"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_identifier)
    chat_id: Mapped[str] = mapped_column(
        ForeignKey("chats.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    turn_title: Mapped[str | None] = mapped_column(String(120))
    rag_run_id: Mapped[str | None] = mapped_column(
        ForeignKey("rag_runs.id", ondelete="SET NULL")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
