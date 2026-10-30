"""Deterministic token-bounded recursive chunking without model dependencies."""

import re

from llm_grounding.domain.corpus import RetrievalUnit
from llm_grounding.domain.evidence import EvidenceUnit
from llm_grounding.ingestion.models import ChunkerConfig
from llm_grounding.knowledge.chunking.common import EvidenceStream, token_spans


class RecursiveTokenChunker:
    """Split text at natural boundaries while enforcing a token ceiling."""

    def __init__(self, config: ChunkerConfig) -> None:
        """Retain validated token sizing and tokenizer settings."""
        self._config = config

    async def chunk(
        self, evidence: tuple[EvidenceUnit, ...], ingestion_id: str, document_name: str
    ) -> tuple[RetrievalUnit, ...]:
        """Return deterministic overlapping chunks with bounded token counts."""
        stream = EvidenceStream.from_evidence(evidence)
        spans = token_spans(stream.text, self._config.tokenizer)
        if not spans:
            return ()
        maximum = self._config.max_tokens or self._config.target_tokens
        boundaries = self._natural_boundaries(stream.text, spans)
        chunks: list[RetrievalUnit] = []
        start = 0
        while start < len(spans):
            target_end = min(start + self._config.target_tokens, len(spans))
            maximum_end = min(start + maximum, len(spans))
            end = self._select_boundary(
                start, target_end, maximum_end, boundaries
            )
            left = spans[start][0]
            right = spans[end - 1][1]
            unit = stream.unit(
                ingestion_id,
                document_name,
                left,
                right,
                len(chunks),
                {
                    "chunking": "recursive_token",
                    "tokenizer": self._config.tokenizer,
                    "token_count": end - start,
                },
            )
            if unit is not None:
                chunks.append(unit)
            if end == len(spans):
                break
            start = max(start + 1, end - self._config.overlap_tokens)
        return tuple(chunks)

    @staticmethod
    def _natural_boundaries(
        text: str, spans: tuple[tuple[int, int], ...]
    ) -> frozenset[int]:
        """Identify token ends after paragraphs, lines, or sentence punctuation."""
        boundaries: set[int] = set()
        for index, (_, end) in enumerate(spans, 1):
            next_start = spans[index][0] if index < len(spans) else len(text)
            gap = text[end:next_start]
            token = text[spans[index - 1][0] : end]
            if "\n" in gap or re.search(r"[.!?。！？][\"'”’)]*$", token):
                boundaries.add(index)
        boundaries.add(len(spans))
        return frozenset(boundaries)

    @staticmethod
    def _select_boundary(
        start: int,
        target_end: int,
        maximum_end: int,
        boundaries: frozenset[int],
    ) -> int:
        """Prefer a natural boundary near the target without exceeding the maximum."""
        candidates = sorted(
            boundary
            for boundary in boundaries
            if target_end <= boundary <= maximum_end
        )
        if candidates:
            return candidates[-1]
        earlier = [
            boundary for boundary in boundaries if start < boundary <= target_end
        ]
        return max(earlier, default=maximum_end)
