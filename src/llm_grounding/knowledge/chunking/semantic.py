"""Embedding-driven semantic boundary detection for canonical evidence streams."""

from collections.abc import Sequence
from dataclasses import replace
from math import sqrt

from llm_grounding.domain.corpus import RetrievalUnit
from llm_grounding.domain.evidence import EvidenceUnit
from llm_grounding.embedding.protocol import Embedder
from llm_grounding.ingestion.models import ChunkerConfig
from llm_grounding.knowledge.chunking.common import EvidenceStream, token_spans
from llm_grounding.knowledge.chunking.recursive_token import RecursiveTokenChunker
from llm_grounding.knowledge.chunking.sentence_window import SentenceWindowChunker


class SemanticChunker:
    """Split sentence groups where adjacent embedding meaning changes sharply."""

    def __init__(self, config: ChunkerConfig, embedder: Embedder) -> None:
        """Retain the configured embedding provider and boundary settings."""
        if config.model_id is None:
            raise ValueError("SemanticChunker requires a model_id")
        self._config = config
        self._embedder = embedder
        self._cache: dict[str, tuple[float, ...]] = {}

    async def chunk(
        self, evidence: tuple[EvidenceUnit, ...], ingestion_id: str, document_name: str
    ) -> tuple[RetrievalUnit, ...]:
        """Create deterministic semantically separated chunks from original text."""
        stream = EvidenceStream.from_evidence(evidence)
        sentences = SentenceWindowChunker._sentences(stream.text)
        if not sentences:
            return ()
        tokens = token_spans(stream.text, self._config.tokenizer)
        if any(
            self._token_count(tokens, start, end) > self._maximum
            for start, end in sentences
        ):
            fallback = await RecursiveTokenChunker(self._config).chunk(
                evidence, ingestion_id, document_name
            )
            return tuple(
                replace(
                    item,
                    metadata={
                        **item.metadata,
                        "chunking": "semantic",
                        "semantic_fallback": "oversized_sentence",
                        "model_id": self._config.model_id,
                    },
                )
                for item in fallback
            )
        texts = [stream.text[start:end] for start, end in sentences]
        values = await self._embed_documents(texts)
        distances = tuple(
            self._distance(left, right)
            for left, right in zip(values, values[1:], strict=False)
        )
        threshold = self._threshold(distances)
        chunks: list[RetrievalUnit] = []
        start_sentence = 0
        start_offset = sentences[0][0]
        for index in range(len(sentences)):
            end_offset = sentences[index][1]
            token_count = self._token_count(tokens, start_offset, end_offset)
            if token_count > self._maximum and index > start_sentence:
                unit = stream.unit(
                    ingestion_id,
                    document_name,
                    start_offset,
                    sentences[index - 1][1],
                    len(chunks),
                    {
                        "chunking": "semantic",
                        "model_id": self._config.model_id,
                        "semantic_threshold": threshold,
                        "token_count": self._token_count(
                            tokens, start_offset, sentences[index - 1][1]
                        ),
                        "sentence_start": start_sentence,
                        "sentence_end": index,
                        "boundary_distance": distances[index - 1]
                        if index - 1 < len(distances)
                        else None,
                        "size_boundary": True,
                    },
                )
                if unit is not None:
                    chunks.append(unit)
                start_sentence = index
                start_offset = sentences[index][0]
                token_count = self._token_count(tokens, start_offset, end_offset)
            distance = distances[index] if index < len(distances) else None
            should_split = (
                index == len(sentences) - 1
                or token_count >= self._maximum
                or (
                    distance is not None
                    and distance >= threshold
                    and token_count >= self._config.minimum_chunk_tokens
                )
            )
            if not should_split:
                continue
            unit = stream.unit(
                ingestion_id,
                document_name,
                start_offset,
                end_offset,
                len(chunks),
                {
                    "chunking": "semantic",
                    "model_id": self._config.model_id,
                    "semantic_threshold": threshold,
                    "token_count": token_count,
                    "sentence_start": start_sentence,
                    "sentence_end": index + 1,
                    "boundary_distance": distance,
                },
            )
            if unit is not None:
                chunks.append(unit)
            start_sentence = index + 1
            if start_sentence < len(sentences):
                start_offset = sentences[start_sentence][0]
        return tuple(chunks)

    @property
    def _maximum(self) -> int:
        """Return the effective maximum semantic chunk size."""
        return self._config.max_tokens or self._config.target_tokens

    async def _embed_documents(self, texts: list[str]) -> tuple[tuple[float, ...], ...]:
        """Embed each sentence once and retain values for repeated chunk calls."""
        missing = [text for text in texts if text not in self._cache]
        if missing:
            vectors = await self._embedder.embed_documents(missing)
            if len(vectors) != len(missing):
                raise RuntimeError(
                    "Semantic embedder returned an unexpected vector count"
                )
            self._cache.update(
                {
                    text: vector.values
                    for text, vector in zip(missing, vectors, strict=True)
                }
            )
        return tuple(self._cache[text] for text in texts)

    def _threshold(self, distances: Sequence[float]) -> float:
        """Resolve an explicit threshold or a deterministic percentile threshold."""
        percentile = self._config.semantic_percentile
        if percentile is None or not distances:
            return self._config.semantic_threshold
        ordered = sorted(distances)
        position = (len(ordered) - 1) * percentile / 100
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = position - lower
        return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction

    @staticmethod
    def _distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
        """Return cosine distance, treating zero vectors as maximally different."""
        if len(left) != len(right) or not left:
            raise ValueError("Semantic embeddings must have equal nonzero dimensions")
        left_norm = sqrt(sum(value * value for value in left))
        right_norm = sqrt(sum(value * value for value in right))
        if left_norm == 0 or right_norm == 0:
            return 1.0
        similarity = sum(a * b for a, b in zip(left, right, strict=True)) / (
            left_norm * right_norm
        )
        return 1.0 - similarity

    @staticmethod
    def _token_count(
        tokens: tuple[tuple[int, int], ...], start: int, end: int
    ) -> int:
        """Count tokenizer spans intersecting one source range."""
        return sum(left < end and right > start for left, right in tokens)
