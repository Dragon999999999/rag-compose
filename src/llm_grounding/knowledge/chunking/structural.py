"""Structure-preserving chunking driven by parser-provided block metadata."""

from typing import Any

from llm_grounding.domain.corpus import RetrievalUnit
from llm_grounding.domain.evidence import EvidenceUnit
from llm_grounding.ingestion.models import ChunkerConfig
from llm_grounding.knowledge.chunking.common import EvidenceStream

_SPECIAL_TYPES = frozenset(
    {"table", "equation", "list", "code", "code_block", "caption", "figure"}
)


class StructuralChunker:
    """Keep structural source blocks intact and attach heading context."""

    def __init__(self, config: ChunkerConfig) -> None:
        """Retain the character target used for paragraph and oversized-block splits."""
        self._config = config

    async def chunk(
        self, evidence: tuple[EvidenceUnit, ...], ingestion_id: str, document_name: str
    ) -> tuple[RetrievalUnit, ...]:
        """Return document-order chunks with structural and heading metadata."""
        stream = EvidenceStream.from_evidence(evidence)
        if not stream.units:
            return ()
        output: list[RetrievalUnit] = []
        heading_stack: list[tuple[int, str]] = []
        paragraph_indices: list[int] = []

        def flush_paragraphs() -> None:
            if not paragraph_indices:
                return
            start_index = 0
            while start_index < len(paragraph_indices):
                end_index = start_index + 1
                while end_index < len(paragraph_indices):
                    candidate_start = stream.spans[paragraph_indices[start_index]][0]
                    candidate_end = stream.spans[paragraph_indices[end_index]][1]
                    if candidate_end - candidate_start > self._config.target_chars:
                        break
                    end_index += 1
                self._emit_span(
                    stream,
                    paragraph_indices[start_index],
                    paragraph_indices[end_index - 1],
                    "paragraph",
                    heading_stack,
                    output,
                    ingestion_id,
                    document_name,
                )
                start_index = end_index
            paragraph_indices.clear()

        for index, item in enumerate(stream.units):
            structure_type, heading_level = self._classify(item)
            if structure_type == "heading":
                flush_paragraphs()
                level = heading_level or 1
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, (item.text or "").strip()))
                self._emit_span(
                    stream,
                    index,
                    index,
                    "heading",
                    heading_stack,
                    output,
                    ingestion_id,
                    document_name,
                    heading_level=level,
                )
            elif structure_type in _SPECIAL_TYPES:
                flush_paragraphs()
                self._emit_span(
                    stream,
                    index,
                    index,
                    structure_type,
                    heading_stack,
                    output,
                    ingestion_id,
                    document_name,
                )
            else:
                paragraph_indices.append(index)
        flush_paragraphs()
        return tuple(output)

    def _emit_span(
        self,
        stream: EvidenceStream,
        first_index: int,
        last_index: int,
        structure_type: str,
        heading_stack: list[tuple[int, str]],
        output: list[RetrievalUnit],
        ingestion_id: str,
        document_name: str,
        *,
        heading_level: int | None = None,
    ) -> None:
        """Emit one block or split an oversized block into character-bounded units."""
        start = stream.spans[first_index][0]
        end = stream.spans[last_index][1]
        cursor = start
        while cursor < end:
            right = min(cursor + self._config.target_chars, end)
            metadata: dict[str, Any] = {
                "chunking": "structural",
                "structure_type": structure_type,
                "heading_hierarchy": [title for _, title in heading_stack],
                "source_block_indices": [
                    stream.units[index].block_index
                    for index in range(first_index, last_index + 1)
                ],
            }
            if heading_level is not None:
                metadata["heading_level"] = heading_level
            unit = stream.unit(
                ingestion_id,
                document_name,
                cursor,
                right,
                len(output),
                metadata,
            )
            if unit is not None:
                output.append(unit)
            cursor = right

    @staticmethod
    def _classify(item: EvidenceUnit) -> tuple[str, int | None]:
        """Read flexible parser metadata without imposing a new parser contract."""
        metadata = item.metadata
        raw_type = metadata.get(
            "structure_type", metadata.get("element_type", metadata.get("kind", ""))
        )
        structure_type = str(raw_type).strip().casefold().replace(" ", "_")
        text = (item.text or "").strip()
        if structure_type in {"heading", "title", "section"} or text.startswith("# "):
            value = metadata.get("heading_level", metadata.get("level", 1))
            try:
                return "heading", max(1, int(str(value)))
            except (TypeError, ValueError):
                return "heading", 1
        if structure_type in _SPECIAL_TYPES:
            return structure_type, None
        return "paragraph", None
