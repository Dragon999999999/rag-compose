"""Transform canonical evidence into retrieval-sized, provenance-mapped units.

Chunkers must preserve every nonempty TEXT character, deterministic ordering, and
evidence offsets. They may duplicate overlap but must not cross document boundaries.
Implementations may use injected embedding or generation providers, but they do not
own persistence or model lifecycle. Late-chunking implementations may additionally
expose the optional ``embed_chunks`` operation used to reuse pooled vectors.
Invalid configuration or mixed-document input must raise ValueError.
"""

from typing import Protocol

from llm_grounding.domain.corpus import RetrievalUnit
from llm_grounding.domain.evidence import EvidenceUnit


class Chunker(Protocol):
    """Derive chunks while keeping model lifecycle outside the chunking boundary."""

    async def chunk(
        self, evidence: tuple[EvidenceUnit, ...], ingestion_id: str, document_name: str
    ) -> tuple[RetrievalUnit, ...]:
        """Return ordered chunks with source evidence mappings and stable IDs."""
        ...
