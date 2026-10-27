"""Local GLiNER metadata enrichment through the shared model manager."""

from contextlib import AbstractAsyncContextManager
from typing import Protocol

from llm_grounding.domain.documents import ParsedDocument
from llm_grounding.domain.evidence import EvidenceUnit
from llm_grounding.infrastructure.config.models import KnowledgeBuilderConfig
from llm_grounding.knowledge.models import KnowledgeBase
from llm_grounding.knowledge.normalization import normalize_parsed_document


class _EntityModel(Protocol):
    """Subset of a local model lease needed by the GLiNER builder."""

    async def extract_entities(
        self, texts: list[str], labels: list[str], threshold: float
    ) -> tuple[tuple[dict[str, object], ...], ...]:
        """Extract labeled source spans from an input batch."""
        ...


class _ModelManager(Protocol):
    """Model-manager boundary used to keep GLiNER lifecycle centralized."""

    def acquire(self, model_id: str) -> AbstractAsyncContextManager[_EntityModel]:
        """Acquire one model lease for a bounded extraction operation."""
        ...


class GlinerMetadataKnowledgeBuilder:
    """Extract configurable named entities with a managed local GLiNER model."""

    def __init__(
        self,
        config: KnowledgeBuilderConfig,
        model_manager: _ModelManager,
    ) -> None:
        """Retain configuration and defer model provisioning until build time."""
        if config.model_id is None:
            raise ValueError("GLiNER metadata enrichment requires model_id")
        if not config.entity_labels:
            raise ValueError("GLiNER metadata enrichment requires entity_labels")
        self._config = config
        self._manager = model_manager

    async def build(self, document: ParsedDocument) -> KnowledgeBase:
        """Enrich source evidence while preserving offsets and original text."""
        base = normalize_parsed_document(document)
        assert self._config.model_id is not None
        labels = list(self._config.entity_labels)
        results: list[tuple[dict[str, object], ...]] = []
        async with self._manager.acquire(self._config.model_id) as model:
            for start in range(0, len(base.evidence), self._config.batch_size):
                batch = base.evidence[start : start + self._config.batch_size]
                extracted = await model.extract_entities(
                    [item.text or "" for item in batch],
                    labels,
                    self._config.confidence_threshold,
                )
                if len(extracted) != len(batch):
                    raise RuntimeError(
                        "GLiNER model returned an unexpected entity batch size"
                    )
                results.extend(extracted)
        return KnowledgeBase(
            tuple(
                self._enrich(item, entities)
                for item, entities in zip(base.evidence, results, strict=True)
            )
        )

    def _enrich(
        self,
        evidence: EvidenceUnit,
        entities: tuple[dict[str, object], ...],
    ) -> EvidenceUnit:
        """Attach source-relative entity spans and model provenance."""
        metadata = {
            **evidence.metadata,
            "knowledge_builder": "metadata_gliner",
            "model_id": self._config.model_id,
            "entity_labels": self._config.entity_labels,
            "confidence_threshold": self._config.confidence_threshold,
            "entities": list(entities),
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
