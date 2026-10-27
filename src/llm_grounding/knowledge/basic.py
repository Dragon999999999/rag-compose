"""Build in-memory text evidence without embeddings, indexing services, or analysis."""

from llm_grounding.domain.documents import ParsedDocument
from llm_grounding.knowledge.models import KnowledgeBase
from llm_grounding.knowledge.normalization import normalize_parsed_document


class BasicKnowledgeBuilder:
    """Map each nonempty parsed text block to stable, provenance-preserving evidence."""

    async def build(self, document: ParsedDocument) -> KnowledgeBase:
        """Copy eligible blocks into evidence; preserve source text and coordinates."""
        return normalize_parsed_document(document)
