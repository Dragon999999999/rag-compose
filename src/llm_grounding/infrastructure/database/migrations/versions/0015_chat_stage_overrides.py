"""Persist validated chat-level component override maps."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0015_chat_stage_overrides"
down_revision = "0014_postgres_fts"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add immutable-by-default JSONB storage for chat override values."""
    op.add_column(
        "chats",
        sa.Column(
            "stage_overrides_json",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )


def downgrade() -> None:
    """Remove persisted chat override values."""
    op.drop_column("chats", "stage_overrides_json")
