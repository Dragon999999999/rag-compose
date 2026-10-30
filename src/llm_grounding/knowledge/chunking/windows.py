"""Shared character-window mapping for fixed and source-block-aware chunking."""

from llm_grounding.domain.corpus import EvidenceReference, RetrievalUnit
from llm_grounding.domain.evidence import EvidenceType, EvidenceUnit, source_order
from llm_grounding.ingestion.models import ChunkerConfig


def create_chunks(
    evidence: tuple[EvidenceUnit, ...],
    ingestion_id: str,
    document_name: str,
    config: ChunkerConfig,
    prefer_blocks: bool,
) -> tuple[RetrievalUnit, ...]:
    """Window a source stream and map intersections back to evidence character spans.

    Block-aware windows prefer the last complete block within the size limit,
    falling back to a fixed window for oversized blocks. Overlap can start inside
    a block; no text is silently discarded to preserve a preferred boundary.
    """
    units = sorted(
        (
            item
            for item in evidence
            if item.evidence_type == EvidenceType.TEXT
            and item.text
            and item.text.strip()
        ),
        key=source_order,
    )
    if not units:
        return ()
    if len({item.document_id for item in units}) != 1:
        raise ValueError("Chunking requires evidence from exactly one document")
    spans: list[tuple[int, int, EvidenceUnit]] = []
    texts: list[str] = []
    offset = 0
    for item in units:
        text = item.text or ""
        spans.append((offset, offset + len(text), item))
        texts.append(text)
        offset += len(text) + 1
    stream = "\n".join(texts)
    chunks: list[RetrievalUnit] = []
    start = 0
    while start < len(stream):
        end = min(start + config.target_chars, len(stream))
        if prefer_blocks and end < len(stream):
            boundaries = [
                right
                for _, right, _ in spans
                if start + config.overlap_chars < right <= end
            ]
            if boundaries:
                end = boundaries[-1]
        contributors = [
            (left, right, item)
            for left, right, item in spans
            if left < end and right > start
        ]
        refs = tuple(
            EvidenceReference(
                item.evidence_id, max(start - left, 0), min(end, right) - left
            )
            for left, right, item in contributors
        )
        pages = [
            item.page_number
            for _, _, item in contributors
            if item.page_number is not None
        ]
        if stream[start:end].strip():
            chunks.append(
                RetrievalUnit(
                    chunk_id=f"{ingestion_id}:c{len(chunks):06d}",
                    document_id=units[0].document_id,
                    ingestion_id=ingestion_id,
                    document_name=document_name,
                    text=stream[start:end],
                    order=len(chunks),
                    page_start=min(pages) if pages else None,
                    page_end=max(pages) if pages else None,
                    evidence_refs=refs,
                    metadata={"sizing": "unicode_characters"},
                )
            )
        if end == len(stream):
            break
        start = end - config.overlap_chars
    return tuple(chunks)
