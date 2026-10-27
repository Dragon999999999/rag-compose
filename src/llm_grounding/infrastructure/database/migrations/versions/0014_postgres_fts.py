"""Add the expression index used by PostgreSQL sparse retrieval."""

import sqlalchemy as sa
from alembic import op

revision = "0014_postgres_fts"
down_revision = "0013_optional_dense_index"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Index existing and future chunk text for the language-neutral FTS path."""
    op.create_index(
        "ix_retrieval_units_text_fts",
        "retrieval_units",
        [sa.text("to_tsvector('simple', text)")],
        postgresql_using="gin",
    )


def downgrade() -> None:
    """Remove the sparse-retrieval expression index."""
    op.drop_index(
        "ix_retrieval_units_text_fts",
        table_name="retrieval_units",
    )
