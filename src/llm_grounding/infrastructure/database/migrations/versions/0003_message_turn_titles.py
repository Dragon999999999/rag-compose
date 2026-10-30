"""Persist navigation labels for prompt-and-answer chat turns."""

import sqlalchemy as sa
from alembic import op

revision = "0003_message_turn_titles"
down_revision = "0002_evidence_geometry"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add an optional generated navigation title to assistant messages."""
    op.add_column("messages", sa.Column("turn_title", sa.String(120), nullable=True))


def downgrade() -> None:
    """Remove persisted navigation labels from chat messages."""
    op.drop_column("messages", "turn_title")
