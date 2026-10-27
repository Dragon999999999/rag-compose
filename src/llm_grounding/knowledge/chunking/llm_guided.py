"""Validated LLM-guided segmentation over original source text."""

import json
from dataclasses import replace
from typing import Protocol

from pydantic import BaseModel, Field

from llm_grounding.domain.corpus import RetrievalUnit
from llm_grounding.domain.evidence import EvidenceUnit
from llm_grounding.ingestion.models import ChunkerConfig
from llm_grounding.knowledge.chunking.common import EvidenceStream, token_spans
from llm_grounding.knowledge.chunking.recursive_token import RecursiveTokenChunker
from llm_grounding.knowledge.chunking.sentence_window import SentenceWindowChunker


class BoundaryGenerator(Protocol):
    """Generate a JSON boundary proposal without rewriting source content."""

    async def generate(self, prompt: str, max_new_tokens: int) -> str:
        """Return the provider response for one segmentation prompt."""
        ...


class BoundaryOutput(BaseModel):
    """Structured boundary response accepted from remote generation providers."""

    boundaries: list[int] = Field(default_factory=list)


class LocalBoundaryGenerator:
    """Generate boundary proposals through one LocalModelManager lease."""

    def __init__(self, model_id: str, manager: object) -> None:
        """Retain the registered model reference without starting its runtime."""
        self._model_id = model_id
        self._manager = manager

    async def generate(self, prompt: str, max_new_tokens: int) -> str:
        """Acquire the local model only for the current generation request."""
        async with self._manager.acquire(self._model_id) as model:
            return await model.generate(prompt, max_new_tokens=max_new_tokens)


class LLMGuidedChunker:
    """Use validated sentence boundary IDs while retaining original source spans."""

    def __init__(self, config: ChunkerConfig, generator: BoundaryGenerator) -> None:
        """Retain the provider and deterministic size constraints."""
        if config.model_id is None:
            raise ValueError("LLMGuidedChunker requires a model_id")
        self._config = config
        self._generator = generator

    async def chunk(
        self, evidence: tuple[EvidenceUnit, ...], ingestion_id: str, document_name: str
    ) -> tuple[RetrievalUnit, ...]:
        """Generate, validate, and apply boundaries, with deterministic fallback."""
        stream = EvidenceStream.from_evidence(evidence)
        sentences = SentenceWindowChunker._sentences(stream.text)
        if not sentences:
            return ()
        tokens = token_spans(stream.text, self._config.tokenizer)
        maximum = self._config.max_tokens or self._config.target_tokens
        if any(
            self._count_tokens(tokens, start, end) > maximum
            for start, end in sentences
        ):
            return await self._fallback(evidence, ingestion_id, document_name)
        prompt = self._prompt(stream.text, sentences)
        raw = await self._generator.generate(prompt, self._config.llm_max_new_tokens)
        boundaries = self._parse_boundaries(raw, len(sentences))
        if boundaries is None:
            return await self._fallback(evidence, ingestion_id, document_name)
        return self._apply_boundaries(
            stream, sentences, boundaries, ingestion_id, document_name
        )

    async def _fallback(
        self,
        evidence: tuple[EvidenceUnit, ...],
        ingestion_id: str,
        document_name: str,
    ) -> tuple[RetrievalUnit, ...]:
        """Apply deterministic token chunking when generation cannot guide safely."""
        fallback = await RecursiveTokenChunker(self._config).chunk(
            evidence, ingestion_id, document_name
        )
        return tuple(
            replace(
                item,
                metadata={
                    **item.metadata,
                    "chunking": "llm_guided",
                    "model_id": self._config.model_id,
                    "boundary_source": "deterministic_fallback",
                },
            )
            for item in fallback
        )

    def _apply_boundaries(
        self,
        stream: EvidenceStream,
        sentences: tuple[tuple[int, int], ...],
        boundaries: frozenset[int],
        ingestion_id: str,
        document_name: str,
    ) -> tuple[RetrievalUnit, ...]:
        """Apply valid sentence IDs while enforcing token limits locally."""
        tokens = token_spans(stream.text, self._config.tokenizer)
        maximum = self._config.max_tokens or self._config.target_tokens
        output: list[RetrievalUnit] = []
        start_index = 0
        for index in range(len(sentences)):
            candidate_end = sentences[index][1]
            candidate_count = self._count_tokens(
                tokens, sentences[start_index][0], candidate_end
            )
            if candidate_count > maximum and index > start_index:
                self._emit(
                    stream,
                    sentences,
                    start_index,
                    index - 1,
                    output,
                    ingestion_id,
                    document_name,
                    boundaries,
                )
                start_index = index
                candidate_count = self._count_tokens(
                    tokens, sentences[start_index][0], candidate_end
                )
            should_split = (
                index == len(sentences) - 1
                or (
                    index in boundaries
                    and candidate_count >= self._config.minimum_chunk_tokens
                )
            )
            if should_split:
                self._emit(
                    stream,
                    sentences,
                    start_index,
                    index,
                    output,
                    ingestion_id,
                    document_name,
                    boundaries,
                )
                start_index = index + 1
        return tuple(output)

    def _emit(
        self,
        stream: EvidenceStream,
        sentences: tuple[tuple[int, int], ...],
        start_index: int,
        end_index: int,
        output: list[RetrievalUnit],
        ingestion_id: str,
        document_name: str,
        boundaries: frozenset[int],
    ) -> None:
        """Emit one source sentence range with validated decision metadata."""
        start, end = sentences[start_index][0], sentences[end_index][1]
        unit = stream.unit(
            ingestion_id,
            document_name,
            start,
            end,
            len(output),
            {
                "chunking": "llm_guided",
                "model_id": self._config.model_id,
                "boundary_source": "llm",
                "boundary_after_sentence": sorted(boundaries),
                "sentence_start": start_index,
                "sentence_end": end_index + 1,
            },
        )
        if unit is not None:
            output.append(unit)

    @staticmethod
    def _prompt(text: str, sentences: tuple[tuple[int, int], ...]) -> str:
        """Create a source-preserving segmentation prompt with stable sentence IDs."""
        items = [
            {"id": index, "text": text[start:end]}
            for index, (start, end) in enumerate(sentences)
        ]
        return (
            "Split the following source sentences into coherent topical segments. "
            "Return only JSON with an integer array named boundaries. Each integer "
            "means split after that sentence ID; never rewrite source text.\n"
            + json.dumps(items, ensure_ascii=False)
        )

    @staticmethod
    def _parse_boundaries(raw: str, sentence_count: int) -> frozenset[int] | None:
        """Parse and validate a JSON boundary array, including fenced responses."""
        candidates = [raw.strip()]
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            candidates.append(raw[start : end + 1])
        for candidate in candidates:
            try:
                value = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            boundaries = value.get("boundaries") if isinstance(value, dict) else None
            if not isinstance(boundaries, list) or not all(
                isinstance(item, int) and not isinstance(item, bool)
                for item in boundaries
            ):
                continue
            if any(item < 0 or item >= sentence_count - 1 for item in boundaries):
                continue
            return frozenset(boundaries)
        return None

    @staticmethod
    def _count_tokens(
        tokens: tuple[tuple[int, int], ...], start: int, end: int
    ) -> int:
        """Count configured tokenizer spans within one sentence range."""
        return sum(left < end and right > start for left, right in tokens)
