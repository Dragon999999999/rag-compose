"""Separate ingestion and query pipeline ownership and index metadata."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0012_pipeline_kinds_and_project_indexes"
down_revision = "0011_pipeline_versions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add pipeline kinds and persist each project's indexed representation."""
    op.add_column("pipelines", sa.Column("kind", sa.String(16), nullable=True))
    op.execute(sa.text("UPDATE pipelines SET kind = 'full' WHERE kind IS NULL"))
    op.alter_column("pipelines", "kind", nullable=False, server_default="full")

    op.add_column(
        "projects",
        sa.Column(
            "ingestion_pipeline_id",
            sa.String(36),
            sa.ForeignKey("pipelines.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.add_column(
        "projects", sa.Column("ingestion_pipeline_version", sa.Integer(), nullable=True)
    )
    op.add_column(
        "projects",
        sa.Column("ingestion_pipeline_config_hash", sa.String(64), nullable=True),
    )
    op.add_column(
        "projects",
        sa.Column("ingestion_representation_hash", sa.String(64), nullable=True),
    )
    op.add_column(
        "projects",
        sa.Column(
            "index_capabilities_json",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )


def downgrade() -> None:
    """Remove project index snapshots and pipeline kind classification."""
    op.drop_column("projects", "index_capabilities_json")
    op.drop_column("projects", "ingestion_representation_hash")
    op.drop_column("projects", "ingestion_pipeline_config_hash")
    op.drop_column("projects", "ingestion_pipeline_version")
    op.drop_column("projects", "ingestion_pipeline_id")
    op.drop_column("pipelines", "kind")
