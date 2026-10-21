"""Persist immutable configuration snapshots for updated pipelines."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0011_pipeline_versions"
down_revision = "0010_chat_pipeline_selection"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create pipeline version snapshots and backfill current configurations."""
    op.create_table(
        "pipeline_versions",
        sa.Column("pipeline_id", sa.String(36), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("config_json", postgresql.JSONB(), nullable=False),
        sa.Column("config_hash", sa.String(64), nullable=False),
        sa.Column("representation_hash", sa.String(64), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now()
        ),
        sa.ForeignKeyConstraint(
            ["pipeline_id"], ["pipelines.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("pipeline_id", "version"),
    )
    op.execute(
        sa.text(
            """
            INSERT INTO pipeline_versions (
                pipeline_id, version, schema_version, description, config_json,
                config_hash, representation_hash
            )
            SELECT id, version, schema_version, description, config_json,
                   config_hash, representation_hash
            FROM pipelines
            """
        )
    )


def downgrade() -> None:
    """Remove immutable pipeline version snapshots."""
    op.drop_table("pipeline_versions")
