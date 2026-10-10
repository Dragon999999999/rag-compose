"""Persist canonical observations for evaluation replay and request recovery."""

import sqlalchemy as sa
from alembic import op

revision = "0006_rag_run_observations"
down_revision = "0005_document_summaries"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create one observation row per RAG run and unique request identity."""
    op.create_table(
        "rag_run_observations",
        sa.Column(
            "run_id",
            sa.String(36),
            sa.ForeignKey("rag_runs.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "request_id", sa.String(255), nullable=False, unique=True, index=True
        ),
        sa.Column("benchmark_case_id", sa.String(255), nullable=True, index=True),
        sa.Column("experiment_id", sa.String(255), nullable=True, index=True),
        sa.Column("status", sa.String(32), nullable=False, index=True),
        sa.Column("observation_json", sa.JSON(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )


def downgrade() -> None:
    """Remove persisted canonical observations."""
    op.drop_table("rag_run_observations")
