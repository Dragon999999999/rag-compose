"""Enable pgvector in the existing PostgreSQL service.

Revision ID: 0017_pgvector_extension
Revises: 0016_component_definitions
Create Date: 2026-10-27
"""

from alembic import op

revision = "0017_pgvector_extension"
down_revision = "0016_component_definitions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Install the vector extension without creating unused index tables."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")


def downgrade() -> None:
    """Leave the extension installed because vector tables may still depend on it."""
