"""Build in-memory text evidence without embeddings, indexing services, or analysis."""

from llm_grounding.domain.documents import ParsedDocument
from llm_grounding.domain.evidence import EvidenceType, EvidenceUnit, source_order
from llm_grounding.knowledge.models import KnowledgeBase


class BasicKnowledgeBuilder:
    """Map each nonempty parsed text block to stable, provenance-preserving evidence."""

    async def build(self, document: ParsedDocument) -> KnowledgeBase:
        """Copy eligible blocks into evidence; preserve source text and coordinates."""
        evidence = [
            EvidenceUnit(
                evidence_id=(
                    f"{element.document_id}:p{element.page_number}"
                    f":b{element.block_index}:text"
                ),
                document_id=element.document_id,
                evidence_type=EvidenceType.TEXT,
                source_uri=document.source.path.resolve().as_uri(),
                order=element.order,
                page_number=element.page_number,
                block_index=element.block_index,
                bounding_box=element.bounding_box,
                text=element.text,
                metadata=element.metadata,
            )
            for element in document.elements
            if element.kind == "text" and element.text.strip()
        ]
        return KnowledgeBase(tuple(sorted(evidence, key=source_order)))
