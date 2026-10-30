"""Normalize parsed source elements into the canonical knowledge contract.

This adapter is the explicit no-knowledge-builder path. It applies the same
normalization used by the basic builder so chunkers, persistence, embedding,
and retrieval receive identical evidence regardless of whether an enrichment
stage was configured.
"""

import json

from llm_grounding.domain.documents import ParsedDocument
from llm_grounding.domain.evidence import EvidenceType, EvidenceUnit, source_order
from llm_grounding.knowledge.models import KnowledgeBase


def normalize_parsed_document(document: ParsedDocument) -> KnowledgeBase:
    """Convert parser output into canonical, provenance-preserving evidence.

    Only nonempty text elements are chunkable by the current chunkers. Other
    element kinds remain available to future knowledge builders through the
    parsed representation but are not silently converted into text evidence.

    Raises:
        ValueError: If parser output violates the document identity, ordering,
            location, or JSON-metadata contract.
    """
    _validate_parsed_document(document)
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


def _validate_parsed_document(document: ParsedDocument) -> None:
    """Validate the parser boundary before constructing canonical evidence."""
    if document.page_count < 0:
        raise ValueError("Parsed document page_count must not be negative")
    if document.source.path is None:
        raise ValueError("Parsed document source path is required")
    try:
        json.dumps(document.metadata)
    except TypeError as exc:
        raise ValueError("Parsed document metadata must be JSON serializable") from exc
    for element in document.elements:
        if element.document_id != document.source.document_id:
            raise ValueError(
                "Parsed document element document_id does not match its source"
            )
        if element.page_number < 1 or element.page_number > document.page_count:
            raise ValueError(
                f"Parsed document element has invalid page_number: "
                f"{element.page_number}"
            )
        if element.block_index < 0 or element.order < 0:
            raise ValueError("Parsed document element locations must not be negative")
        try:
            json.dumps(element.metadata)
        except TypeError as exc:
            raise ValueError(
                f"Metadata for parsed element {element.order} must be JSON serializable"
            ) from exc
