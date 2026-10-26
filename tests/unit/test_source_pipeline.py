"""Verify parsing provenance and immutable knowledge."""

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pymupdf
import pytest

from llm_grounding.domain.documents import DocumentSource
from llm_grounding.domain.evidence import EvidenceType
from llm_grounding.knowledge.basic import BasicKnowledgeBuilder
from llm_grounding.parsing.pymupdf import PyMuPDFDocumentParser


async def test_parser_retains_pages_blocks_boxes_and_source_order(
    pdf_path: Path,
) -> None:
    """Real PDF extraction preserves structure while ordering blocks by position."""
    source = DocumentSource("paper", pdf_path)
    parsed = await PyMuPDFDocumentParser().parse(source)
    assert parsed.source == source
    assert parsed.page_count == 2
    assert len(parsed.elements) == 3
    assert [element.page_number for element in parsed.elements] == [1, 1, 2]
    assert [element.order for element in parsed.elements] == [0, 1, 2]
    assert parsed.elements[0].text.startswith("Upper")
    assert parsed.elements[1].text.startswith("Lower")
    assert [element.block_index for element in parsed.elements[:2]] == [1, 0]
    assert all(element.bounding_box is not None for element in parsed.elements)
    assert len(parsed.metadata["sha256"]) == 64


async def test_parser_groups_nearby_rows_and_filters_nontext(
    pdf_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Within a ten-point row band, x position takes priority over tiny y offsets."""
    blocks = [
        (200, 71, 300, 80, "right", 0, 0),
        (20, 79, 100, 89, "left", 1, 0),
        (0, 0, 10, 10, "image metadata", 2, 1),
        (0, 0, 10, 10, "  \n", 3, 0),
    ]
    document = MagicMock()
    document.__enter__.return_value = document
    document.is_pdf = True
    document.needs_pass = False
    document.__len__.return_value = 1
    document.__iter__.return_value = iter([SimpleNamespace(get_text=lambda _: blocks)])
    monkeypatch.setattr(pymupdf, "open", lambda **kwargs: document)
    parsed = await PyMuPDFDocumentParser().parse(DocumentSource("paper", pdf_path))
    assert [element.text for element in parsed.elements] == ["left", "right"]
    assert parsed.elements[0].bounding_box == (20, 79, 100, 89)


@pytest.mark.parametrize(
    "kind", ["missing", "directory", "nonpdf", "corrupt", "empty", "blank"]
)
async def test_parser_rejects_invalid_or_textless_sources(
    tmp_path: Path, kind: str
) -> None:
    """Missing, malformed, and scanned/blank inputs fail before model generation."""
    path = tmp_path / "input.pdf"
    if kind == "directory":
        path.mkdir()
    elif kind == "nonpdf":
        path = tmp_path / "input.txt"
        path.write_text("text", encoding="utf-8")
    elif kind == "corrupt":
        path.write_bytes(b"not a PDF")
    elif kind == "empty":
        path.write_bytes(b"")
    elif kind == "blank":
        with pymupdf.open() as document:
            document.new_page()
            document.save(path)
    with pytest.raises((ValueError, FileNotFoundError)):
        await PyMuPDFDocumentParser().parse(DocumentSource("paper", path))


async def test_knowledge_retains_provenance_and_stable_immutable_evidence(
    pdf_path: Path,
) -> None:
    """Repeated construction preserves IDs and detaches metadata from source state."""
    parsed = await PyMuPDFDocumentParser().parse(DocumentSource("paper", pdf_path))
    builder = BasicKnowledgeBuilder()
    knowledge = await builder.build(parsed)
    assert knowledge == await builder.build(parsed)
    for element, evidence in zip(parsed.elements, knowledge.evidence, strict=True):
        assert evidence.document_id == element.document_id
        assert evidence.page_number == element.page_number
        assert evidence.block_index == element.block_index
        assert evidence.bounding_box == element.bounding_box
        assert evidence.text == element.text
        assert evidence.source_uri == pdf_path.as_uri()
        assert evidence.evidence_type == EvidenceType.TEXT
        assert not hasattr(evidence, "score")
    with pytest.raises(FrozenInstanceError):
        knowledge.evidence[0].text = "changed"
    with pytest.raises(TypeError):
        knowledge.evidence[0].metadata["changed"] = "yes"


async def test_knowledge_skips_nontext_and_empty_elements(pdf_path: Path) -> None:
    """Baseline knowledge constructs only eligible source text."""
    parsed = await PyMuPDFDocumentParser().parse(DocumentSource("paper", pdf_path))
    text = parsed.elements[0]
    parsed = replace(
        parsed, elements=(text, replace(text, kind="figure"), replace(text, text=" "))
    )
    knowledge = await BasicKnowledgeBuilder().build(parsed)
    assert len(knowledge.evidence) == 1
