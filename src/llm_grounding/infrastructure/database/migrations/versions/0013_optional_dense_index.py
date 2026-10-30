"""Allow PostgreSQL-only ingestion records without vector metadata.

Revision ID: 0013_optional_dense_index
Revises: 0012_pipeline_kinds_and_project_indexes
Create Date: 2026-10-27
"""

from alembic import op

revision = "0013_optional_dense_index"
down_revision = "0012_pipeline_kinds_and_project_indexes"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Allow an ingestion to persist without an embedding or vector index."""
    op.alter_column("document_ingestions", "embedding_hash", nullable=True)
    op.alter_column("document_ingestions", "dimension", nullable=True)


def downgrade() -> None:
    """Restore dense metadata requirements after checking existing rows."""
    op.alter_column("document_ingestions", "dimension", nullable=False)
    op.alter_column("document_ingestions", "embedding_hash", nullable=False)
