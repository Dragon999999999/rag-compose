"""Persist PDF evidence geometry for browser citation navigation."""

import sqlalchemy as sa
from alembic import op

revision = "0002_evidence_geometry"
down_revision = "0001_phase_one"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add optional parser-provided PDF bounding boxes to canonical evidence."""
    op.add_column("evidence_units", sa.Column("bounding_box", sa.JSON(), nullable=True))


def downgrade() -> None:
    """Remove optional PDF geometry while retaining canonical evidence text."""
    op.drop_column("evidence_units", "bounding_box")
