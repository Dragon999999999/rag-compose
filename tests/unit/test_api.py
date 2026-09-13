"""Test deterministic browser provenance DTO serialization."""

from datetime import UTC, datetime

from llm_grounding.api.main import _evidence_dto, _item, _message_dto
from llm_grounding.domain.corpus import Chat, EvidenceReference, Message, RetrievalUnit
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


def test_message_dto_exposes_persisted_turn_title() -> None:
    """The browser receives the generated navigation label with its answer."""
    message = Message(
        "message",
        "chat",
        "assistant",
        "Answer text",
        datetime.now(UTC),
        "run",
        "Paper contribution",
    )

    dto = _message_dto(message)

    assert dto["turn_title"] == "Paper contribution"


def test_chat_dto_exposes_automatic_name_state() -> None:
    """The browser can distinguish a placeholder from a finalized chat name."""
    chat = Chat("chat", "project", "New chat", datetime.now(UTC), False)

    dto = _item(chat)

    assert dto["name"] == "New chat"
    assert dto["chat_name_set"] is False
