"""Persist conversation pipeline selections and run-level pipeline provenance."""

import sqlalchemy as sa
from alembic import op

revision = "0010_chat_pipeline_selection"
down_revision = "0009_pipelines"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add nullable selection snapshots for existing conversations and runs."""
    op.add_column(
        "chats",
        sa.Column(
            "pipeline_id",
            sa.String(36),
            sa.ForeignKey("pipelines.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.add_column("chats", sa.Column("pipeline_version", sa.Integer(), nullable=True))
    op.add_column(
        "chats", sa.Column("pipeline_config_hash", sa.String(64), nullable=True)
    )
    op.add_column(
        "chats",
        sa.Column("pipeline_representation_hash", sa.String(64), nullable=True),
    )
    op.add_column(
        "rag_runs",
        sa.Column(
            "pipeline_id",
            sa.String(36),
            sa.ForeignKey("pipelines.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.add_column(
        "rag_runs", sa.Column("pipeline_version", sa.Integer(), nullable=True)
    )
    op.add_column(
        "rag_runs", sa.Column("pipeline_config_hash", sa.String(64), nullable=True)
    )
    op.add_column(
        "rag_runs",
        sa.Column("pipeline_representation_hash", sa.String(64), nullable=True),
    )


def downgrade() -> None:
    """Remove conversation selections and explicit run provenance columns."""
    op.drop_column("rag_runs", "pipeline_representation_hash")
    op.drop_column("rag_runs", "pipeline_config_hash")
    op.drop_column("rag_runs", "pipeline_version")
    op.drop_column("rag_runs", "pipeline_id")
    op.drop_column("chats", "pipeline_representation_hash")
    op.drop_column("chats", "pipeline_config_hash")
    op.drop_column("chats", "pipeline_version")
    op.drop_column("chats", "pipeline_id")
