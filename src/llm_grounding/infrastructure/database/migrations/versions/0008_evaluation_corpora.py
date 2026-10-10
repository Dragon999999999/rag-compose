"""Create isolated evaluation corpus inventories."""

import sqlalchemy as sa
from alembic import op

revision = "0008_evaluation_corpora"
down_revision = "0007_ingestion_observations"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create benchmark corpus storage independent from production projects."""
    op.create_table(
        "evaluation_corpora",
        sa.Column("id", sa.String(255), primary_key=True),
        sa.Column("parent_corpus_id", sa.String(255), nullable=True),
        sa.Column("corpus_hash", sa.String(64), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("documents_json", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("chunks_json", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column(
            "perturbation_metadata", sa.JSON(), nullable=False, server_default="{}"
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_table(
        "evaluation_corpus_documents",
        sa.Column("id", sa.String(255), primary_key=True),
        sa.Column(
            "corpus_id",
            sa.String(255),
            sa.ForeignKey("evaluation_corpora.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("document_json", sa.JSON(), nullable=False),
    )
    op.create_table(
        "evaluation_corpus_chunks",
        sa.Column("id", sa.String(255), primary_key=True),
        sa.Column(
            "corpus_id",
            sa.String(255),
            sa.ForeignKey("evaluation_corpora.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("chunk_json", sa.JSON(), nullable=False),
    )
    op.create_table(
        "evaluation_corpus_operations",
        sa.Column("id", sa.String(255), primary_key=True),
        sa.Column("corpus_id", sa.String(255), nullable=False, index=True),
        sa.Column("status", sa.String(32), nullable=False, index=True),
        sa.Column("operation_json", sa.JSON(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now()
        ),
    )


def downgrade() -> None:
    """Remove isolated evaluation corpus storage."""
    op.drop_table("evaluation_corpus_operations")
    op.drop_table("evaluation_corpus_chunks")
    op.drop_table("evaluation_corpus_documents")
    op.drop_table("evaluation_corpora")
