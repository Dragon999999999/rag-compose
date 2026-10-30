"""Fixed Unicode-character windows with exact source span mappings."""

from llm_grounding.domain.corpus import RetrievalUnit
from llm_grounding.domain.evidence import EvidenceUnit
from llm_grounding.ingestion.models import ChunkerConfig
from llm_grounding.knowledge.chunking.windows import create_chunks


class FixedWindowChunker:
    """Split source text by character count with configurable overlap."""

    def __init__(self, config: ChunkerConfig) -> None:
        """Retain validated sizing configuration."""
        self._config = config

    async def chunk(
        self, evidence: tuple[EvidenceUnit, ...], ingestion_id: str, document_name: str
    ) -> tuple[RetrievalUnit, ...]:
        """Return fixed windows without losing oversized source blocks."""
        return create_chunks(evidence, ingestion_id, document_name, self._config, False)
