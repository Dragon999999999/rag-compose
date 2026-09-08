"""SQLAlchemy persistence models; domain objects are mapped at repository boundaries.

PostgreSQL is canonical for projects, document metadata, evidence/chunks, chats,
messages, and run metadata. Original PDFs remain in artifact storage; embeddings
are derived and held by pgvector. The schema intentionally omits advanced claims.
"""

from datetime import datetime
from uuid import uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
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
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class DocumentRow(Base):
    """Persist deduplicated immutable PDF metadata and artifact reference."""

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
    embedding_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    error: Mapped[str | None] = mapped_column(Text)
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
    text: Mapped[str | None] = mapped_column(Text)
    metadata_json: Mapped[dict[str, str]] = mapped_column(JSON, default=dict)


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
    metadata_json: Mapped[dict[str, str]] = mapped_column(JSON, default=dict)


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


class RetrievalEmbeddingRow(Base):
    """Persist pgvector's rebuildable derived vector for one indexed project chunk."""

    __tablename__ = "retrieval_embeddings"
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True
    )
    chunk_id: Mapped[str] = mapped_column(
        ForeignKey("retrieval_units.id", ondelete="CASCADE"), primary_key=True
    )
    ingestion_id: Mapped[str] = mapped_column(
        ForeignKey("document_ingestions.id", ondelete="CASCADE"), nullable=False
    )
    model_identity: Mapped[str] = mapped_column(String(128), nullable=False)
    dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(), nullable=False)


class ChatRow(Base):
    """Persist a conversation that consumes one project corpus."""

    __tablename__ = "chats"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_identifier)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class RagRunRow(Base):
    """Persist run identity/status/metadata while detailed trace remains application-owned."""

    __tablename__ = "rag_runs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    project_id: Mapped[str | None] = mapped_column(
        ForeignKey("projects.id", ondelete="SET NULL")
    )
    architecture_name: Mapped[str] = mapped_column(String(255), nullable=False)
    architecture_version: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    metadata_json: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
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
    rag_run_id: Mapped[str | None] = mapped_column(
        ForeignKey("rag_runs.id", ondelete="SET NULL")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
