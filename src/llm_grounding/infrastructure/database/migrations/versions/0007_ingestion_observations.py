"""Persist evidence and chunk inventories for ingestion diagnostics."""

import sqlalchemy as sa
from alembic import op

revision = "0007_ingestion_observations"
down_revision = "0006_rag_run_observations"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create one durable inventory record for each document ingestion."""
    op.create_table(
        "ingestion_observations",
        sa.Column(
            "ingestion_id",
            sa.String(36),
            sa.ForeignKey("document_ingestions.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("observation_json", sa.JSON(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )


def downgrade() -> None:
    """Remove persisted ingestion inventories."""
    op.drop_table("ingestion_observations")
