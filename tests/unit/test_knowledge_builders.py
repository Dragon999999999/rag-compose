"""Verify structural and metadata knowledge-builder contracts."""

from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import Any

from llm_grounding.domain.documents import (
    DocumentElement,
    DocumentSource,
    ParsedDocument,
)
from llm_grounding.domain.evidence import EvidenceType
from llm_grounding.infrastructure.config.models import KnowledgeBuilderConfig
from llm_grounding.knowledge.metadata_gliner import GlinerMetadataKnowledgeBuilder
from llm_grounding.knowledge.metadata_rules import RuleMetadataKnowledgeBuilder
from llm_grounding.knowledge.metadata_spacy import SpacyMetadataKnowledgeBuilder
from llm_grounding.knowledge.structural import StructuralKnowledgeBuilder


def _document() -> ParsedDocument:
    """Create parser output containing headings, text, and a table block."""
    source = DocumentSource("document-1", Path("document.pdf"))
    return ParsedDocument(
        source=source,
        page_count=1,
        elements=(
            DocumentElement(
                "document-1", 1, 0, 0, "Methods", kind="heading", metadata={"level": 1}
            ),
            DocumentElement(
                "document-1", 1, 1, 1, "Alice measured 12 kg in EXP-42.", kind="text"
            ),
            DocumentElement(
                "document-1", 1, 2, 2, "A | B\n1 | 2", kind="table"
            ),
        ),
    )


async def test_structural_builder_preserves_blocks_and_parent_relationships() -> None:
    """Structural output retains source text and links descendants to headings."""
    result = await StructuralKnowledgeBuilder().build(_document())

    assert [item.text for item in result.evidence] == [
        "Methods",
        "Alice measured 12 kg in EXP-42.",
        "A | B\n1 | 2",
    ]
    assert result.evidence[2].evidence_type == EvidenceType.TABLE
    assert (
        result.evidence[1].metadata["parent_evidence_id"]
        == result.evidence[0].evidence_id
    )
    assert result.evidence[0].metadata["child_evidence_ids"] == [
        result.evidence[1].evidence_id,
        result.evidence[2].evidence_id,
    ]


async def test_rule_builder_extracts_values_and_source_offsets() -> None:
    """Rules record deterministic values and offsets without rewriting text."""
    config = KnowledgeBuilderConfig(
        implementation="metadata_rules",
        max_keywords=4,
        keyword_min_length=4,
    )
    result = await RuleMetadataKnowledgeBuilder(config).build(_document())
    evidence = result.evidence[0]
    extractions = evidence.metadata["metadata_extractions"]

    assert evidence.text == "Alice measured 12 kg in EXP-42."
    assert any(
        item["type"] == "unit" and item["value"] == "12 kg"
        for item in extractions
    )
    assert any(
        item["type"] == "identifier" and item["value"] == "EXP-42"
        for item in extractions
    )
    assert evidence.metadata["keyword_extractions"]


class _FakeSpacyPipeline:
    """Minimal spaCy-compatible pipeline for deterministic builder tests."""

    def pipe(self, texts: list[str], batch_size: int) -> list[Any]:
        """Return one entity-bearing fake document per input text."""
        del batch_size
        return [_FakeSpacyDocument(text) for text in texts]


class _FakeSpacyDocument:
    """Small document object exposing the spaCy attributes used by the builder."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.ents = (
            type(
                "Entity",
                (),
                {
                    "text": "Alice",
                    "label_": "PERSON",
                    "start_char": 0,
                    "end_char": 5,
                },
            )(),
        )
        self.lang_ = "en"
        self.sents: tuple[object, ...] = ()

    def __len__(self) -> int:
        """Return a deterministic token-count approximation."""
        return len(self.text.split())


async def test_spacy_builder_records_filtered_entity_spans() -> None:
    """SpaCy entities retain labels and source-relative offsets."""
    config = KnowledgeBuilderConfig(
        implementation="metadata_spacy",
        spacy_entity_labels=("PERSON",),
    )
    result = await SpacyMetadataKnowledgeBuilder(config, _FakeSpacyPipeline()).build(
        _document()
    )

    assert result.evidence[0].metadata["entities"] == [
        {"text": "Alice", "label": "PERSON", "start": 0, "end": 5}
    ]


class _FakeEntityModel:
    """Return one deterministic GLiNER entity for every input."""

    async def extract_entities(
        self, texts: list[str], labels: list[str], threshold: float
    ) -> tuple[tuple[dict[str, object], ...], ...]:
        """Return source-relative entity records using the requested settings."""
        assert labels == ["person"]
        assert threshold == 0.5
        return tuple(
            (
                {
                    "text": text.split()[0],
                    "label": "person",
                    "start": 0,
                    "end": len(text.split()[0]),
                    "score": 0.9,
                },
            )
            for text in texts
        )


class _FakeLease(AbstractAsyncContextManager[_FakeEntityModel]):
    """Async context manager representing one model-manager lease."""

    async def __aenter__(self) -> _FakeEntityModel:
        """Return the fake model."""
        return _FakeEntityModel()

    async def __aexit__(self, *args: object) -> None:
        """Release the fake model lease."""


class _FakeManager:
    """Provide the manager acquire boundary without downloading a model."""

    def acquire(self, model_id: str) -> _FakeLease:
        """Return a fake lease for the expected registered model ID."""
        assert model_id == "gliner_small"
        return _FakeLease()


async def test_gliner_builder_uses_managed_model_and_preserves_provenance() -> None:
    """GLiNER results are attached to canonical evidence with model metadata."""
    config = KnowledgeBuilderConfig(
        implementation="metadata_gliner",
        model_id="gliner_small",
        entity_labels=("person",),
        confidence_threshold=0.5,
    )
    result = await GlinerMetadataKnowledgeBuilder(config, _FakeManager()).build(
        _document()
    )

    assert result.evidence[0].text == "Alice measured 12 kg in EXP-42."
    assert result.evidence[0].metadata["model_id"] == "gliner_small"
    assert result.evidence[0].metadata["entities"][0]["start"] == 0
