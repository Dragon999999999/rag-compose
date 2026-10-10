"""Map runtime retrieval results to canonical provenance records."""

from collections.abc import Mapping
from typing import Any

from llm_grounding.domain.corpus import ChunkRetrievalResult
from llm_grounding.domain.evidence import EvidenceUnit
from llm_grounding.observability.models import RetrievedItem, Score, SourceRef


def retrieved_item(
    result: ChunkRetrievalResult,
    evidence: Mapping[str, EvidenceUnit] | None = None,
) -> RetrievedItem:
    """Create a stable observation item without fabricating source coordinates.

    Character offsets are retained only when a chunk has one unambiguous evidence
    reference.  Multi-reference chunks retain their evidence IDs in metadata and
    leave document-relative offsets unavailable.
    """
    chunk = result.chunk
    evidence = evidence or {}
    references = chunk.evidence_refs
    first = evidence.get(references[0].evidence_id) if references else None
    offsets = (
        (references[0].start, references[0].end)
        if len(references) == 1
        else (None, None)
    )
    metadata: dict[str, Any] = dict(chunk.metadata)
    metadata["retriever_id"] = result.retriever_id
    if references:
        metadata["evidence_ids"] = [reference.evidence_id for reference in references]
    return RetrievedItem(
        retrieval_id=chunk.chunk_id,
        rank=result.rank,
        text=chunk.text,
        source=SourceRef(
            document_id=chunk.document_id,
            chunk_id=chunk.chunk_id,
            ingestion_id=chunk.ingestion_id,
            page=first.page_number if first else chunk.page_start,
            start_char=offsets[0],
            end_char=offsets[1],
            metadata={
                "document_name": chunk.document_name,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "evidence_geometry": (
                    first.bounding_box if first is not None else None
                ),
            },
        ),
        score=Score(
            value=result.score,
            type="cosine_similarity" if result.score is not None else None,
        ),
        metadata=metadata,
    )


def retrieved_items(
    results: tuple[ChunkRetrievalResult, ...],
    evidence: Mapping[str, EvidenceUnit] | None = None,
) -> list[RetrievedItem]:
    """Map an ordered result set while preserving ranks and stable identities."""
    return [retrieved_item(result, evidence) for result in results]
