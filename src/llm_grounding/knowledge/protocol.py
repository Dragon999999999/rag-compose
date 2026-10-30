"""Transform parsed sources into canonical evidence consumed by retrieval.

Builders must preserve provenance and stable evidence IDs, without query scores.
Implementations may construct persistent indexes and perform I/O; failures must
propagate instead of silently dropping source material. Input parsing is complete
before this boundary. See docs/architecture.md.
"""

from typing import Protocol

from llm_grounding.domain.documents import ParsedDocument
from llm_grounding.knowledge.models import KnowledgeBase


class KnowledgeBuilder(Protocol):
    """Define the interchangeable knowledge component contract."""

    async def build(self, document: ParsedDocument) -> KnowledgeBase:
        """Build knowledge while preserving source provenance."""
        ...
