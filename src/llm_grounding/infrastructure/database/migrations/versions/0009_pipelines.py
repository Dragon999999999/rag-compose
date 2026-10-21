"""Persist versioned RAG pipeline configurations."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0009_pipelines"
down_revision = "0008_evaluation_corpora"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create the PostgreSQL pipeline source-of-truth table."""
    op.create_table(
        "pipelines",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column(
            "config_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False
        ),
        sa.Column("config_hash", sa.String(64), nullable=False),
        sa.Column("representation_hash", sa.String(64), nullable=False),
        sa.Column("source_kind", sa.String(16), nullable=False),
        sa.Column("source_path", sa.Text(), nullable=True),
        sa.Column("source_fingerprint", sa.String(64), nullable=True),
        sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column(
            "is_default", sa.Boolean(), nullable=False, server_default=sa.false()
        ),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()
        ),
        sa.UniqueConstraint("name", name="uq_pipeline_name"),
    )


def downgrade() -> None:
    """Remove persisted pipeline definitions."""
    op.drop_table("pipelines")
