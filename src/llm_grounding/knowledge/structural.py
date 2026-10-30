"""Reconstruct parser-provided document structure as canonical evidence."""

from collections.abc import Mapping

from llm_grounding.domain.documents import DocumentElement, ParsedDocument
from llm_grounding.domain.evidence import EvidenceType, EvidenceUnit
from llm_grounding.knowledge.models import KnowledgeBase
from llm_grounding.knowledge.normalization import normalize_parsed_document

_EVIDENCE_TYPES = {
    "table": EvidenceType.TABLE,
    "figure": EvidenceType.FIGURE,
    "equation": EvidenceType.EQUATION,
    "caption": EvidenceType.CAPTION,
}


class StructuralKnowledgeBuilder:
    """Preserve parser blocks while attaching hierarchy and parent relationships."""

    async def build(self, document: ParsedDocument) -> KnowledgeBase:
        """Build ordered evidence from all parser elements without altering text."""
        # Reuse the basic validator so structural handling cannot weaken the parser
        # boundary or change the established behavior for malformed documents.
        normalize_parsed_document(document)
        source_uri = document.source.path.resolve().as_uri()
        stack: list[tuple[int, str, str]] = []
        evidence: list[EvidenceUnit] = []
        children: dict[str, list[str]] = {}
        for element in sorted(document.elements, key=lambda item: item.order):
            kind, heading_level = self._kind_and_level(element)
            evidence_type = _EVIDENCE_TYPES.get(kind, EvidenceType.TEXT)
            evidence_id = self._evidence_id(element, evidence_type)
            if kind == "heading":
                level = heading_level or 1
                while stack and stack[-1][0] >= level:
                    stack.pop()
                parent_id = stack[-1][1] if stack else None
                stack.append((level, evidence_id, element.text.strip()))
            else:
                parent_id = stack[-1][1] if stack else None
            section_path = [item[2] for item in stack]
            metadata: dict[str, object] = dict(element.metadata)
            metadata.update(
                {
                    "knowledge_builder": "structural",
                    "structure_type": kind,
                    "section_path": section_path,
                    "parent_evidence_id": parent_id,
                    "source_order": element.order,
                }
            )
            if heading_level is not None:
                metadata["heading_level"] = heading_level
            unit = EvidenceUnit(
                evidence_id=evidence_id,
                document_id=element.document_id,
                evidence_type=evidence_type,
                source_uri=source_uri,
                order=element.order,
                page_number=element.page_number,
                block_index=element.block_index,
                bounding_box=element.bounding_box,
                text=element.text,
                metadata=metadata,
            )
            evidence.append(unit)
            if parent_id is not None:
                children.setdefault(parent_id, []).append(evidence_id)
        enriched = tuple(
            EvidenceUnit(
                evidence_id=item.evidence_id,
                document_id=item.document_id,
                evidence_type=item.evidence_type,
                source_uri=item.source_uri,
                order=item.order,
                page_number=item.page_number,
                block_index=item.block_index,
                bounding_box=item.bounding_box,
                text=item.text,
                metadata={
                    **item.metadata,
                    "child_evidence_ids": list(children.get(item.evidence_id, ())),
                },
            )
            for item in evidence
        )
        return KnowledgeBase(enriched)

    @staticmethod
    def _evidence_id(element: DocumentElement, evidence_type: EvidenceType) -> str:
        """Create a stable source-block identity compatible with basic text IDs."""
        return (
            f"{element.document_id}:p{element.page_number}:b{element.block_index}:"
            f"{evidence_type.value}"
        )

    @staticmethod
    def _kind_and_level(element: DocumentElement) -> tuple[str, int | None]:
        """Resolve flexible parser metadata without requiring a new parser format."""
        metadata: Mapping[str, object] = element.metadata
        raw_kind = metadata.get("structure_type", metadata.get("element_type"))
        kind = str(raw_kind or element.kind or "text").casefold().replace("-", "_")
        if kind.startswith("heading"):
            suffix = kind.removeprefix("heading").strip("_ ")
            level = int(suffix) if suffix.isdigit() else None
            return "heading", level or StructuralKnowledgeBuilder._level(metadata)
        return kind, StructuralKnowledgeBuilder._level(metadata)

    @staticmethod
    def _level(metadata: Mapping[str, object]) -> int | None:
        """Read a positive heading level when parser metadata provides one."""
        value = metadata.get("heading_level", metadata.get("level"))
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
        return None
