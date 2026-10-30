"""Persist grounded document summaries independently from chats."""

import sqlalchemy as sa
from alembic import op

revision = "0005_document_summaries"
down_revision = "0004_chat_name_state"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create the document-summary lifecycle and its linked RAG run reference."""
    op.create_table(
        "document_summaries",
        sa.Column(
            "document_id",
            sa.String(36),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "ingestion_id",
            sa.String(36),
            sa.ForeignKey("document_ingestions.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column(
            "rag_run_id",
            sa.String(36),
            sa.ForeignKey("rag_runs.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("page_count", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint(
            "document_id", "ingestion_id", name="uq_document_summary_ingestion"
        ),
    )


def downgrade() -> None:
    """Remove persisted document-summary state."""
    op.drop_table("document_summaries")
