"""Deterministic evidence assessors used by query pipelines.

These assessors consume the already retrieved context and its native scores.
They do not call models, mutate evidence, reorder candidates, or consult
benchmark answers.  A result with status ``unavailable`` means the configured
measurement could not be made from the information available at query time.
"""

import math
import re
from collections.abc import Mapping
from statistics import mean, pstdev

from llm_grounding.domain.answers import Assessment, JsonValue, ReasoningResult
from llm_grounding.domain.corpus import (
    ChunkRetrievalResult,
    CorpusRequest,
    GroundingContext,
    RetrievalUnit,
)
from llm_grounding.domain.evidence import RetrievalResult

_TOKEN = re.compile(r"[\w]+", re.UNICODE)


def _json_strings(values: list[str] | set[str]) -> list[JsonValue]:
    """Convert string collections to the recursive JSON value type."""
    return [value for value in values]


def _context_from_evidence(
    evidence: tuple[RetrievalResult, ...], query: str = "query"
) -> GroundingContext:
    """Adapt the legacy evidence-only assessor call to canonical context."""
    chunks = tuple(
        ChunkRetrievalResult(
            RetrievalUnit(
                item.evidence.evidence_id,
                item.evidence.document_id,
                "unknown",
                item.evidence.source_uri,
                item.evidence.text or "",
                item.evidence.order,
                item.evidence.page_number,
                item.evidence.page_number,
                (),
                dict(item.evidence.metadata),
            ),
            item.rank,
            item.retriever_id,
            item.score,
            item.ranking_score,
            item.rerank_score,
        )
        for item in evidence
    )
    return GroundingContext(CorpusRequest("unknown", query), chunks)


def _tokens(value: str) -> set[str]:
    """Return normalized lexical terms for deterministic coverage metrics."""
    return {token.casefold() for token in _TOKEN.findall(value)}


def _status_assessment(
    result_id: str,
    assessor: str,
    score: float | None,
    status: str,
    evidence_ids: tuple[str, ...] = (),
    metadata: Mapping[str, JsonValue] | None = None,
) -> Assessment:
    """Build the shared assessment envelope used by deterministic assessors."""
    return Assessment(
        result_id,
        assessor,
        assessor,
        score,
        status,
        evidence_ids,
        dict(metadata or {}),
        status,
    )


class NoOpAssessor:
    """Preserve baseline behavior without producing measurements."""

    async def assess_context(
        self, result: ReasoningResult, context: GroundingContext
    ) -> tuple[Assessment, ...]:
        """Return no assessment and perform no context processing."""
        del result, context
        return ()

    async def assess(
        self, result: ReasoningResult, evidence: tuple[RetrievalResult, ...]
    ) -> tuple[Assessment, ...]:
        """Provide the protocol-compatible no-op entry point."""
        del result, evidence
        return ()


class RetrievalConfidenceAssessor:
    """Summarize native retrieval scores without treating them as probabilities."""

    def __init__(
        self,
        *,
        threshold: float | None = None,
        score_kind: str | None = None,
        min_threshold_count: int = 0,
    ) -> None:
        """Configure an optional score-kind-specific threshold."""
        if min_threshold_count < 0:
            raise ValueError("min_threshold_count must not be negative")
        self._threshold = threshold
        self._score_kind = score_kind
        self._min_threshold_count = min_threshold_count

    async def assess_context(
        self, result: ReasoningResult, context: GroundingContext
    ) -> tuple[Assessment, ...]:
        """Return maximum, mean, spread, margin, and threshold statistics."""
        scored = [
            item
            for item in context.chunks
            if item.ranking_score is not None
            and (
                self._score_kind is None
                or item.ranking_score.kind == self._score_kind
            )
        ]
        if not scored:
            return (
                _status_assessment(
                    result.result_id,
                    "retrieval_confidence",
                    None,
                    "unavailable",
                    metadata={"reason": "no_compatible_retrieval_scores"},
                ),
            )
        values = [item.ranking_score.value for item in scored if item.ranking_score]
        direction = scored[0].ranking_score.higher_is_better  # type: ignore[union-attr]
        ordered = sorted(values, reverse=direction)
        margin = ordered[0] - ordered[1] if len(ordered) > 1 else None
        threshold_count = (
            sum(
                value >= self._threshold if direction else value <= self._threshold
                for value in values
            )
            if self._threshold is not None
            else None
        )
        status = "uncertain"
        if self._threshold is not None:
            status = (
                "passed"
                if threshold_count is not None
                and threshold_count >= self._min_threshold_count
                else "failed"
            )
        return (
            _status_assessment(
                result.result_id,
                "retrieval_confidence",
                max(values) if direction else min(values),
                status,
                tuple(item.chunk.chunk_id for item in scored),
                {
                    "score_kind": scored[0].ranking_score.kind,  # type: ignore[union-attr]
                    "method_id": scored[0].ranking_score.method_id,  # type: ignore[union-attr]
                    "maximum": max(values),
                    "mean": mean(values),
                    "standard_deviation": pstdev(values),
                    "top_score_margin": margin,
                    "threshold": self._threshold,
                    "threshold_count": threshold_count,
                    "calibrated": any(
                        item.ranking_score.calibrated
                        for item in scored
                        if item.ranking_score
                    ),
                },
            ),
        )

    async def assess(
        self, result: ReasoningResult, evidence: tuple[RetrievalResult, ...]
    ) -> tuple[Assessment, ...]:
        """Support callers that provide no canonical context envelope."""
        return await self.assess_context(result, _context_from_evidence(evidence))


class RelevanceAssessor:
    """Assess existing reranker scores, optionally falling back to retrieval scores."""

    def __init__(
        self,
        *,
        threshold: float | None = None,
        fallback_to_retrieval: bool = False,
    ) -> None:
        """Configure score selection and an optional native-score threshold."""
        self._threshold = threshold
        self._fallback = fallback_to_retrieval

    async def assess_context(
        self, result: ReasoningResult, context: GroundingContext
    ) -> tuple[Assessment, ...]:
        """Measure reranker output without invoking another relevance model."""
        values: list[tuple[str, float, str]] = []
        for item in context.chunks:
            score = item.rerank_score
            if score is not None:
                values.append((item.chunk.chunk_id, score.value, score.kind))
            elif self._fallback and item.ranking_score is not None:
                values.append(
                    (
                        item.chunk.chunk_id,
                        item.ranking_score.value,
                        item.ranking_score.kind,
                    )
                )
        if not values:
            return (
                _status_assessment(
                    result.result_id,
                    "relevance",
                    None,
                    "unavailable",
                    metadata={"reason": "no_rerank_score"},
                ),
            )
        scores = [score for _, score, _ in values]
        status = "uncertain" if self._threshold is None else (
            "passed" if mean(scores) >= self._threshold else "failed"
        )
        return (
            _status_assessment(
                result.result_id,
                "relevance",
                mean(scores),
                status,
                tuple(chunk_id for chunk_id, _, _ in values),
                {
                    "score_kinds": _json_strings(
                        sorted({kind for _, _, kind in values})
                    ),
                    "minimum": min(scores),
                    "maximum": max(scores),
                    "mean": mean(scores),
                    "threshold": self._threshold,
                    "fallback_to_retrieval": self._fallback,
                },
            ),
        )

    async def assess(
        self, result: ReasoningResult, evidence: tuple[RetrievalResult, ...]
    ) -> tuple[Assessment, ...]:
        """Report that canonical context is required for score-aware assessment."""
        return await self.assess_context(result, _context_from_evidence(evidence))


class EvidenceSufficiencyAssessor:
    """Estimate query-term, entity, and requested-aspect coverage heuristically."""

    def __init__(
        self,
        *,
        requested_aspects: tuple[str, ...] = (),
        threshold: float = 0.5,
    ) -> None:
        """Configure requested aspects and the coverage threshold."""
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between zero and one")
        self._aspects = requested_aspects
        self._threshold = threshold

    async def assess_context(
        self, result: ReasoningResult, context: GroundingContext
    ) -> tuple[Assessment, ...]:
        """Return transparent lexical/entity/aspect coverage estimates."""
        query_terms = _tokens(context.request.task)
        text_terms = set().union(*(_tokens(item.chunk.text) for item in context.chunks))
        covered_terms = sorted(query_terms & text_terms)
        entities = self._metadata_terms(context, ("entities", "named_entities"))
        query_entities = {
            term for term in query_terms if term in entities
        }
        covered_aspects = [
            aspect for aspect in self._aspects if _tokens(aspect) & text_terms
        ]
        missing_aspects = [
            aspect for aspect in self._aspects if aspect not in covered_aspects
        ]
        ratios = [
            len(covered_terms) / len(query_terms) if query_terms else 0.0,
            len(query_entities) / len(query_terms) if query_terms else 0.0,
            len(covered_aspects) / len(self._aspects) if self._aspects else 1.0,
        ]
        score = mean(ratios)
        return (
            _status_assessment(
                result.result_id,
                "evidence_sufficiency",
                score,
                "passed" if score >= self._threshold else "failed",
                tuple(item.chunk.chunk_id for item in context.chunks),
                {
                    "heuristic": True,
                    "covered_query_terms": _json_strings(covered_terms),
                    "missing_query_terms": _json_strings(
                        sorted(query_terms - text_terms)
                    ),
                    "covered_entities": _json_strings(sorted(query_entities)),
                    "covered_aspects": _json_strings(covered_aspects),
                    "missing_aspects": _json_strings(missing_aspects),
                    "threshold": self._threshold,
                    "warning": "heuristic estimate; not verified answerability",
                },
            ),
        )

    @staticmethod
    def _metadata_terms(context: GroundingContext, keys: tuple[str, ...]) -> set[str]:
        """Extract entity-like strings from builder metadata without assumptions."""
        values: list[str] = []
        for item in context.chunks:
            for key in keys:
                raw = item.chunk.metadata.get(key)
                if isinstance(raw, list):
                    values.extend(str(value) for value in raw)
                elif isinstance(raw, str):
                    values.append(raw)
        return set().union(*(_tokens(value) for value in values)) if values else set()

    async def assess(
        self, result: ReasoningResult, evidence: tuple[RetrievalResult, ...]
    ) -> tuple[Assessment, ...]:
        """Return unavailable when only the unstructured protocol input is supplied."""
        return await self.assess_context(result, _context_from_evidence(evidence))


class DiversityAssessor:
    """Measure lexical or supplied-vector redundancy without changing result order."""

    def __init__(
        self,
        *,
        max_pairs: int = 1000,
        embeddings: Mapping[str, tuple[float, ...]] | None = None,
    ) -> None:
        """Configure bounded pair comparisons and optional precomputed vectors."""
        if max_pairs < 1:
            raise ValueError("max_pairs must be positive")
        self._max_pairs = max_pairs
        self._embeddings = embeddings or {}

    async def assess_context(
        self, result: ReasoningResult, context: GroundingContext
    ) -> tuple[Assessment, ...]:
        """Return one-minus-average pairwise redundancy and comparison statistics."""
        items = context.chunks
        pairs: list[float] = []
        vector_pairs = 0
        for left_index, left in enumerate(items):
            for right in items[left_index + 1 :]:
                if len(pairs) >= self._max_pairs:
                    break
                left_vector = self._embeddings.get(left.chunk.chunk_id)
                right_vector = self._embeddings.get(right.chunk.chunk_id)
                if left_vector is not None and right_vector is not None:
                    similarity = _cosine(left_vector, right_vector)
                    vector_pairs += 1
                else:
                    first, second = _tokens(left.chunk.text), _tokens(right.chunk.text)
                    union = first | second
                    similarity = len(first & second) / len(union) if union else 1.0
                pairs.append(similarity)
            if len(pairs) >= self._max_pairs:
                break
        if not items:
            status, score = "unavailable", None
        else:
            score = 1.0 - mean(pairs) if pairs else 1.0
            status = "passed"
        return (
            _status_assessment(
                result.result_id,
                "diversity",
                score,
                status,
                tuple(item.chunk.chunk_id for item in items),
                {
                    "pair_count": len(pairs),
                    "average_redundancy": 1.0 - score if score is not None else None,
                    "vector_pair_count": vector_pairs,
                    "lexical_pair_count": len(pairs) - vector_pairs,
                    "max_pairs": self._max_pairs,
                },
            ),
        )

    async def assess(
        self, result: ReasoningResult, evidence: tuple[RetrievalResult, ...]
    ) -> tuple[Assessment, ...]:
        """Return unavailable when the canonical context envelope is absent."""
        return await self.assess_context(result, _context_from_evidence(evidence))


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    """Compare compatible precomputed vectors without loading an embedder."""
    if len(left) != len(right):
        raise ValueError("Diversity vectors must use the same embedding dimension")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (
        left_norm * right_norm
    )
