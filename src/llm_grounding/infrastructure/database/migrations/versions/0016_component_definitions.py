"""Persist validated standalone component definitions."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0016_component_definitions"
down_revision = "0015_chat_stage_overrides"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create the PostgreSQL component-definition registry table."""
    op.create_table(
        "component_definitions",
        sa.Column("id", sa.String(length=255), nullable=False),
        sa.Column("category", sa.String(length=64), nullable=False),
        sa.Column("type", sa.String(length=64), nullable=False),
        sa.Column("implementation", sa.String(length=128), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("config_json", postgresql.JSONB(), nullable=False),
        sa.Column(
            "requirements_json", postgresql.JSONB(), nullable=False
        ),
        sa.Column("source_path", sa.Text(), nullable=False),
        sa.Column("source_fingerprint", sa.String(length=64), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    """Drop the standalone component-definition registry table."""
    op.drop_table("component_definitions")
