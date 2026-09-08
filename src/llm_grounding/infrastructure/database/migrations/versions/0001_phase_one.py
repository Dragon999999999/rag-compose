"""Create Phase 1 canonical corpus schema and derived pgvector index.

Revision ID: 0001_phase_one
Revises:
Create Date: 2026-09-08
"""

import sqlalchemy as sa
from alembic import op

revision = "0001_phase_one"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create canonical metadata tables, provenance mappings, and rebuildable vector rows."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "projects",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_table(
        "documents",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(1024), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False, unique=True),
        sa.Column("artifact_key", sa.String(1024), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_table(
        "document_ingestions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "document_id",
            sa.String(36),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("config_hash", sa.String(64), nullable=False),
        sa.Column("config_json", sa.Text(), nullable=False),
        sa.Column("embedding_hash", sa.String(64), nullable=False),
        sa.Column("dimension", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("error", sa.Text()),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.UniqueConstraint(
            "document_id", "config_hash", name="uq_document_ingestion_config"
        ),
    )
    op.create_table(
        "project_documents",
        sa.Column(
            "project_id",
            sa.String(36),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "document_id",
            sa.String(36),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "ingestion_id",
            sa.String(36),
            sa.ForeignKey("document_ingestions.id", ondelete="SET NULL"),
        ),
    )
    op.create_table(
        "evidence_units",
        sa.Column("id", sa.String(255), primary_key=True),
        sa.Column(
            "ingestion_id",
            sa.String(36),
            sa.ForeignKey("document_ingestions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "document_id",
            sa.String(36),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("evidence_type", sa.String(16), nullable=False),
        sa.Column("source_uri", sa.Text(), nullable=False),
        sa.Column("ordering", sa.Integer(), nullable=False),
        sa.Column("page_number", sa.Integer()),
        sa.Column("block_index", sa.Integer()),
        sa.Column("text", sa.Text()),
        sa.Column(
            "metadata_json",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'::json"),
        ),
    )
    op.create_table(
        "retrieval_units",
        sa.Column("id", sa.String(255), primary_key=True),
        sa.Column(
            "ingestion_id",
            sa.String(36),
            sa.ForeignKey("document_ingestions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "document_id",
            sa.String(36),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("document_name", sa.String(1024), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("ordering", sa.Integer(), nullable=False),
        sa.Column("page_start", sa.Integer()),
        sa.Column("page_end", sa.Integer()),
        sa.Column(
            "metadata_json",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'::json"),
        ),
    )
    op.create_table(
        "retrieval_unit_members",
        sa.Column(
            "chunk_id",
            sa.String(255),
            sa.ForeignKey("retrieval_units.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "evidence_id",
            sa.String(255),
            sa.ForeignKey("evidence_units.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("start", sa.Integer(), primary_key=True),
        sa.Column("end", sa.Integer(), nullable=False),
    )
    op.execute(
        "CREATE TABLE retrieval_embeddings ("
        "project_id varchar(36) REFERENCES projects(id) ON DELETE CASCADE, "
        "chunk_id varchar(255) REFERENCES retrieval_units(id) ON DELETE CASCADE, "
        "ingestion_id varchar(36) REFERENCES document_ingestions(id) ON DELETE CASCADE NOT NULL, "
        "model_identity varchar(128) NOT NULL, dimension integer NOT NULL, "
        "embedding vector NOT NULL, PRIMARY KEY (project_id, chunk_id))"
    )
    op.create_index(
        "ix_retrieval_embeddings_project_model",
        "retrieval_embeddings",
        ["project_id", "model_identity"],
    )
    op.create_table(
        "chats",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "project_id",
            sa.String(36),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_table(
        "rag_runs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "project_id",
            sa.String(36),
            sa.ForeignKey("projects.id", ondelete="SET NULL"),
        ),
        sa.Column("architecture_name", sa.String(255), nullable=False),
        sa.Column("architecture_version", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column(
            "metadata_json",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'::json"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_table(
        "messages",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "chat_id",
            sa.String(36),
            sa.ForeignKey("chats.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("role", sa.String(16), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column(
            "rag_run_id",
            sa.String(36),
            sa.ForeignKey("rag_runs.id", ondelete="SET NULL"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )


def downgrade() -> None:
    """Drop Phase 1 tables; artifact files intentionally remain outside migrations."""
    for table in (
        "messages",
        "rag_runs",
        "chats",
        "retrieval_embeddings",
        "retrieval_unit_members",
        "retrieval_units",
        "evidence_units",
        "project_documents",
        "document_ingestions",
        "documents",
        "projects",
    ):
        op.drop_table(table)
