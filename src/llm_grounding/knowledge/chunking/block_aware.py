"""Prefer source-block boundaries while bounding character-sized retrieval units."""

from llm_grounding.domain.corpus import RetrievalUnit
from llm_grounding.domain.evidence import EvidenceUnit
from llm_grounding.ingestion.models import ChunkerConfig
from llm_grounding.knowledge.chunking.windows import create_chunks


class BlockAwareChunker:
    """Accumulate adjacent blocks, splitting oversized blocks with fixed fallback."""

    def __init__(self, config: ChunkerConfig) -> None:
        """Retain validated sizing configuration."""
        self._config = config

    async def chunk(
        self, evidence: tuple[EvidenceUnit, ...], ingestion_id: str, document_name: str
    ) -> tuple[RetrievalUnit, ...]:
        """Prefer whole blocks and preserve overlap provenance."""
        return create_chunks(evidence, ingestion_id, document_name, self._config, True)
