"""Transform canonical evidence into retrieval-sized, provenance-mapped units.

Chunkers must preserve every nonempty TEXT character, deterministic ordering, and
evidence offsets. They may duplicate overlap but must not cross document boundaries.
Implementations are pure transformations and perform no embedding or persistence.
Invalid configuration or mixed-document input must raise ValueError.
"""

from typing import Protocol

from llm_grounding.domain.corpus import RetrievalUnit
from llm_grounding.domain.evidence import EvidenceUnit


class Chunker(Protocol):
    """Derive chunks independently of parsers and embedding providers."""

    async def chunk(
        self, evidence: tuple[EvidenceUnit, ...], ingestion_id: str, document_name: str
    ) -> tuple[RetrievalUnit, ...]:
        """Return ordered chunks with source evidence mappings and stable IDs."""
        ...
