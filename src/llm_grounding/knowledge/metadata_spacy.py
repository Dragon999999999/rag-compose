"""Optional local spaCy metadata enrichment for canonical evidence."""

from typing import Any

from llm_grounding.domain.documents import ParsedDocument
from llm_grounding.domain.evidence import EvidenceUnit
from llm_grounding.infrastructure.config.models import KnowledgeBuilderConfig
from llm_grounding.knowledge.models import KnowledgeBase
from llm_grounding.knowledge.normalization import normalize_parsed_document


class SpacyMetadataKnowledgeBuilder:
    """Extract configured named entities using a lazily loaded spaCy pipeline."""

    def __init__(self, config: KnowledgeBuilderConfig, nlp: Any | None = None) -> None:
        """Store model settings; importing or loading spaCy is deferred to build."""
        self._config = config
        self._nlp = nlp

    async def build(self, document: ParsedDocument) -> KnowledgeBase:
        """Enrich canonical text evidence with entity spans and optional linguistics."""
        base = normalize_parsed_document(document)
        nlp = self._pipeline()
        texts = [item.text or "" for item in base.evidence]
        docs = tuple(nlp.pipe(texts, batch_size=self._config.batch_size))
        return KnowledgeBase(
            tuple(
                self._enrich(item, parsed)
                for item, parsed in zip(base.evidence, docs, strict=True)
            )
        )

    def _pipeline(self) -> Any:
        """Load the configured model with an actionable dependency error."""
        if self._nlp is not None:
            return self._nlp
        try:
            import spacy  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "spaCy metadata enrichment requires the optional spaCy dependency"
            ) from exc
        try:
            self._nlp = spacy.load(self._config.spacy_model)
        except OSError as exc:
            raise RuntimeError(
                f"spaCy model {self._config.spacy_model!r} is unavailable; "
                "install the configured model before selecting this builder"
            ) from exc
        return self._nlp

    def _enrich(self, evidence: EvidenceUnit, doc: Any) -> EvidenceUnit:
        """Copy source evidence while adding source-relative spaCy metadata."""
        allowed = set(self._config.spacy_entity_labels)
        entities = [
            {
                "text": entity.text,
                "label": entity.label_,
                "start": entity.start_char,
                "end": entity.end_char,
            }
            for entity in doc.ents
            if not allowed or entity.label_ in allowed
        ]
        metadata: dict[str, object] = {
            **evidence.metadata,
            "knowledge_builder": "metadata_spacy",
            "spacy_model": self._config.spacy_model,
            "entities": entities,
        }
        if self._config.include_linguistic_metadata:
            metadata["linguistic"] = {
                "language": getattr(doc, "lang_", None),
                "token_count": len(doc),
                "sentence_count": sum(1 for _ in doc.sents),
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
