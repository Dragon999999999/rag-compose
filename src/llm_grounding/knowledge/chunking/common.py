"""Shared provenance helpers for deterministic evidence-based chunkers."""

from dataclasses import dataclass
from typing import Any

from llm_grounding.domain.corpus import EvidenceReference, RetrievalUnit
from llm_grounding.domain.evidence import EvidenceType, EvidenceUnit, source_order


@dataclass(frozen=True)
class EvidenceStream:
    """Represent one document's text evidence as a newline-separated stream."""

    units: tuple[EvidenceUnit, ...]
    spans: tuple[tuple[int, int, EvidenceUnit], ...]
    text: str

    @classmethod
    def from_evidence(cls, evidence: tuple[EvidenceUnit, ...]) -> "EvidenceStream":
        """Order nonempty text evidence while validating document identity."""
        units = tuple(
            sorted(
                (
                    item
                    for item in evidence
                    if item.evidence_type == EvidenceType.TEXT
                    and item.text
                    and item.text.strip()
                ),
                key=source_order,
            )
        )
        if not units:
            return cls((), (), "")
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
        return cls(units, tuple(spans), "\n".join(texts))

    def unit(
        self,
        ingestion_id: str,
        document_name: str,
        start: int,
        end: int,
        order: int,
        metadata: dict[str, Any] | None = None,
        *,
        chunk_id: str | None = None,
        text_start: int | None = None,
        text_end: int | None = None,
    ) -> RetrievalUnit | None:
        """Create one retrieval unit and map its indexed span to source evidence."""
        if start < 0 or end <= start or end > len(self.text):
            raise ValueError("Chunk span is outside the evidence stream")
        indexed_start = start if text_start is None else text_start
        indexed_end = end if text_end is None else text_end
        if indexed_end <= indexed_start:
            return None
        contributors = [
            (left, right, item)
            for left, right, item in self.spans
            if left < indexed_end and right > indexed_start
        ]
        refs = tuple(
            EvidenceReference(
                item.evidence_id,
                max(indexed_start - left, 0),
                min(indexed_end, right) - left,
            )
            for left, right, item in contributors
        )
        pages = [
            item.page_number
            for _, _, item in contributors
            if item.page_number is not None
        ]
        return RetrievalUnit(
            chunk_id=chunk_id or f"{ingestion_id}:c{order:06d}",
            document_id=self.units[0].document_id,
            ingestion_id=ingestion_id,
            document_name=document_name,
            text=self.text[indexed_start:indexed_end],
            order=order,
            page_start=min(pages) if pages else None,
            page_end=max(pages) if pages else None,
            evidence_refs=refs,
            metadata=dict(metadata or {}),
        )


def token_spans(text: str, tokenizer: str) -> tuple[tuple[int, int], ...]:
    """Return deterministic Unicode-aware token spans for the supported modes."""
    import re

    pattern = r"\S+" if tokenizer == "whitespace" else r"\w+|[^\w\s]"
    return tuple((match.start(), match.end()) for match in re.finditer(pattern, text))


def split_token_ranges(
    spans: tuple[tuple[int, int], ...],
    target: int,
    maximum: int,
    overlap: int,
) -> tuple[tuple[int, int], ...]:
    """Split token spans with bounded overlap and guaranteed forward progress."""
    if not spans:
        return ()
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < len(spans):
        target_end = min(start + target, len(spans))
        maximum_end = min(start + maximum, len(spans))
        end = target_end
        if target_end < maximum_end:
            end = maximum_end
        ranges.append((start, end))
        if end == len(spans):
            break
        next_start = max(start + 1, end - overlap)
        start = next_start
    return tuple(ranges)
