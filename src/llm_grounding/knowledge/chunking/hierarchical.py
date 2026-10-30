"""Deterministic parent-child token hierarchies for retrieval and context expansion."""

from llm_grounding.domain.corpus import RetrievalUnit
from llm_grounding.domain.evidence import EvidenceUnit
from llm_grounding.ingestion.models import ChunkerConfig
from llm_grounding.knowledge.chunking.common import (
    EvidenceStream,
    split_token_ranges,
    token_spans,
)


class HierarchicalChunker:
    """Emit configured parent and child chunks linked through metadata IDs."""

    def __init__(self, config: ChunkerConfig) -> None:
        """Retain validated hierarchy levels and overlap settings."""
        self._config = config

    async def chunk(
        self, evidence: tuple[EvidenceUnit, ...], ingestion_id: str, document_name: str
    ) -> tuple[RetrievalUnit, ...]:
        """Return all hierarchy levels in deterministic parent-before-child order."""
        stream = EvidenceStream.from_evidence(evidence)
        spans = token_spans(stream.text, self._config.tokenizer)
        if not spans:
            return ()
        output: list[RetrievalUnit] = []
        parents: tuple[tuple[int, int, str | None, tuple[str, ...]], ...] = (
            (0, len(spans), None, ()),
        )
        for level_index, level_size in enumerate(self._config.hierarchical_levels):
            children: list[tuple[int, int, str, tuple[str, ...]]] = []
            for parent_start, parent_end, parent_id, path in parents:
                local_spans = spans[parent_start:parent_end]
                ranges = split_token_ranges(
                    local_spans,
                    level_size,
                    level_size,
                    self._config.overlap_tokens,
                )
                for local_start, local_end in ranges:
                    start = parent_start + local_start
                    end = parent_start + local_end
                    chunk_id = f"{ingestion_id}:h{level_index}:{len(output):06d}"
                    current_path = (*path, chunk_id)
                    unit = stream.unit(
                        ingestion_id,
                        document_name,
                        spans[start][0],
                        spans[end - 1][1],
                        len(output),
                        {
                            "chunking": "hierarchical",
                            "chunk_level": level_index,
                            "token_count": end - start,
                            "tokenizer": self._config.tokenizer,
                            "parent_chunk_id": parent_id,
                            "hierarchy_path": list(current_path),
                        },
                        chunk_id=chunk_id,
                    )
                    if unit is not None:
                        output.append(unit)
                        children.append((start, end, chunk_id, current_path))
            parents = tuple(children)
        return tuple(output)
