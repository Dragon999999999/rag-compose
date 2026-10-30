"""Sentence indexing with separately stored surrounding context windows."""

import re

from llm_grounding.domain.corpus import RetrievalUnit
from llm_grounding.domain.evidence import EvidenceUnit
from llm_grounding.ingestion.models import ChunkerConfig
from llm_grounding.knowledge.chunking.common import EvidenceStream


class SentenceWindowChunker:
    """Index sentence groups while retaining configurable neighboring context."""

    def __init__(self, config: ChunkerConfig) -> None:
        """Retain sentence grouping and context-window settings."""
        self._config = config

    async def chunk(
        self, evidence: tuple[EvidenceUnit, ...], ingestion_id: str, document_name: str
    ) -> tuple[RetrievalUnit, ...]:
        """Return indexed sentence groups with context stored in metadata."""
        stream = EvidenceStream.from_evidence(evidence)
        sentences = self._sentences(stream.text)
        output: list[RetrievalUnit] = []
        group_size = self._config.sentences_per_chunk
        for group_start in range(0, len(sentences), group_size):
            group_end = min(group_start + group_size, len(sentences))
            indexed_start = sentences[group_start][0]
            indexed_end = sentences[group_end - 1][1]
            context_start_index = max(0, group_start - self._config.sentence_window)
            context_end_index = min(
                len(sentences), group_end + self._config.sentence_window
            )
            context_start = sentences[context_start_index][0]
            context_end = sentences[context_end_index - 1][1]
            unit = stream.unit(
                ingestion_id,
                document_name,
                context_start,
                context_end,
                len(output),
                {
                    "chunking": "sentence_window",
                    "sentence_start": group_start,
                    "sentence_end": group_end,
                    "sentence_start_offset": indexed_start,
                    "sentence_end_offset": indexed_end,
                    "context_sentence_start": context_start_index,
                    "context_sentence_end": context_end_index,
                    "context_start_offset": context_start,
                    "context_end_offset": context_end,
                    "context_text": stream.text[context_start:context_end],
                },
                text_start=indexed_start,
                text_end=indexed_end,
            )
            if unit is not None:
                output.append(unit)
        return tuple(output)

    @staticmethod
    def _sentences(text: str) -> tuple[tuple[int, int], ...]:
        """Segment common Latin and CJK sentence punctuation without external models."""
        if not text.strip():
            return ()
        boundary = re.compile(r"[.!?]+[\"'”’\)\]]*(?=\s|$)|[。！？]+|\n{2,}")
        spans: list[tuple[int, int]] = []
        start = 0
        for match in boundary.finditer(text):
            end = match.end()
            left = start + len(text[start:end]) - len(text[start:end].lstrip())
            right = end - len(text[start:end]) + len(text[start:end].rstrip())
            if right > left:
                spans.append((left, right))
            start = end
        tail = text[start:]
        left = start + len(tail) - len(tail.lstrip())
        right = len(text) - len(tail) + len(tail.rstrip())
        if right > left:
            spans.append((left, right))
        return tuple(spans)
