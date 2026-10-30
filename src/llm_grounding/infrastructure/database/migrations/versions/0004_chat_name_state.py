"""Persist whether a chat name may still be generated automatically."""

import sqlalchemy as sa
from alembic import op

revision = "0004_chat_name_state"
down_revision = "0003_message_turn_titles"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add chat-name state and preserve existing user-selected names."""
    op.add_column(
        "chats",
        sa.Column(
            "chat_name_set",
            sa.Boolean(),
            server_default=sa.false(),
            nullable=False,
        ),
    )
    op.execute(
        sa.text(
            "UPDATE chats SET chat_name_set = true "
            "WHERE lower(trim(name)) <> 'new chat'"
        )
    )
    op.alter_column("chats", "chat_name_set", server_default=None)


def downgrade() -> None:
    """Remove persisted automatic chat-name state."""
    op.drop_column("chats", "chat_name_set")
