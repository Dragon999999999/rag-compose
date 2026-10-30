"""Deterministic metadata extraction over canonical source evidence."""

import re
from collections import Counter

from llm_grounding.domain.documents import ParsedDocument
from llm_grounding.domain.evidence import EvidenceUnit
from llm_grounding.infrastructure.config.models import KnowledgeBuilderConfig
from llm_grounding.knowledge.models import KnowledgeBase
from llm_grounding.knowledge.normalization import normalize_parsed_document

_DEFAULT_PATTERNS = {
    "date": r"\b(?:\d{1,4}[-/.]){2}\d{1,4}\b|\b(?:19|20)\d{2}\b",
    "number": r"(?<!\w)[+-]?\d+(?:[.,]\d+)?(?:[eE][+-]?\d+)?(?!\w)",
    "unit": (
        r"(?<!\w)[+-]?\d+(?:[.,]\d+)?\s?"
        r"(?:%|°[CF]|km/h|kg|mg|g|mm|cm|m|s|ms|Hz|kHz|MHz|Pa|kPa|V|A|W)\b"
    ),
    "identifier": r"\b[A-Z][A-Z0-9]*(?:[-_][A-Z0-9]+)+\b|\b[A-Z]{2,}\d[A-Z0-9]*\b",
}
_DEFAULT_STOPWORDS = frozenset(
    "a an and are as at be by for from in is it of on or that the this to was with"
    .split()
)


class RuleMetadataKnowledgeBuilder:
    """Enrich text evidence with reproducible regex and keyword extractions."""

    def __init__(self, config: KnowledgeBuilderConfig | None = None) -> None:
        """Compile configured rules once while keeping extraction deterministic."""
        config = config or KnowledgeBuilderConfig(implementation="metadata_rules")
        patterns = {**_DEFAULT_PATTERNS, **config.rule_patterns}
        try:
            self._patterns = {
                name: re.compile(pattern, re.IGNORECASE | re.UNICODE)
                for name, pattern in patterns.items()
            }
        except re.error as exc:
            raise ValueError("Invalid metadata rule regular expression") from exc
        self._max_keywords = config.max_keywords
        self._keyword_min_length = config.keyword_min_length
        self._stopwords = _DEFAULT_STOPWORDS | {
            word.casefold() for word in config.stopwords
        }

    async def build(self, document: ParsedDocument) -> KnowledgeBase:
        """Return basic evidence with offset-bearing deterministic metadata."""
        base = normalize_parsed_document(document)
        return KnowledgeBase(tuple(self._enrich(item) for item in base.evidence))

    def _enrich(self, evidence: EvidenceUnit) -> EvidenceUnit:
        """Attach regex matches and ranked lexical keywords to one evidence unit."""
        text = evidence.text or ""
        extractions = [
            {
                "type": name,
                "value": match.group(0),
                "start": match.start(),
                "end": match.end(),
            }
            for name, pattern in self._patterns.items()
            for match in pattern.finditer(text)
        ]
        keywords = self._keywords(text)
        metadata = {
            **evidence.metadata,
            "knowledge_builder": "metadata_rules",
            "metadata_extractions": extractions,
            "keywords": [item["value"] for item in keywords],
            "keyword_extractions": keywords,
        }
        return EvidenceUnit(
            evidence_id=evidence.evidence_id,
            document_id=evidence.document_id,
            evidence_type=evidence.evidence_type,
            source_uri=evidence.source_uri,
            order=evidence.order,
            page_number=evidence.page_number,
            block_index=evidence.block_index,
            bounding_box=evidence.bounding_box,
            text=evidence.text,
            metadata=metadata,
        )

    def _keywords(self, text: str) -> list[dict[str, object]]:
        """Rank repeated content words and retain the first source occurrence."""
        matches = tuple(re.finditer(r"[\wÀ-ÖØ-öø-ÿ'-]+", text, re.UNICODE))
        counts = Counter(
            match.group(0).casefold()
            for match in matches
            if len(match.group(0)) >= self._keyword_min_length
            and match.group(0).casefold() not in self._stopwords
        )
        ranked = sorted(counts, key=lambda value: (-counts[value], value))[
            : self._max_keywords
        ]
        first_match = {
            match.group(0).casefold(): match
            for match in matches
            if match.group(0).casefold() in ranked
        }
        return [
            {
                "value": first_match[word].group(0),
                "start": first_match[word].start(),
                "end": first_match[word].end(),
                "count": counts[word],
            }
            for word in ranked
        ]
