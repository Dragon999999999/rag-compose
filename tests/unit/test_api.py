"""Test deterministic browser provenance DTO serialization."""

from llm_grounding.api.main import _evidence_dto
from llm_grounding.domain.corpus import EvidenceReference, RetrievalUnit
from llm_grounding.domain.evidence import EvidenceType, EvidenceUnit


def test_evidence_dto_exposes_canonical_geometry_without_model_identifiers() -> None:
    """A browser citation is derived from persisted chunk/evidence provenance."""
    chunk = RetrievalUnit(
        "canonical-chunk",
        "document",
        "ingestion",
        "paper.pdf",
        "Supported source text.",
        0,
        2,
        2,
        (EvidenceReference("evidence", 0, 22),),
    )
    evidence = EvidenceUnit(
        "evidence",
        "document",
        EvidenceType.TEXT,
        "file:///paper.pdf",
        0,
        2,
        0,
        (10.0, 20.0, 110.0, 40.0),
        "Supported source text.",
    )

    dto = _evidence_dto(chunk, {"evidence": evidence})

    assert dto["chunk_id"] == "canonical-chunk"
    assert dto["document_name"] == "paper.pdf"
    assert dto["locations"] == [
        {
            "page": 2,
            "bounding_boxes": [{"x": 10.0, "y": 20.0, "width": 100.0, "height": 20.0}],
            "positioning": "exact",
        }
    ]
