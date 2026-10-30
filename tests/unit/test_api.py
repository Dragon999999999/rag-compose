"""Test deterministic browser provenance DTO serialization."""

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace

import httpx
import pytest

from llm_grounding.api.main import (
    _claim_source_chunk_ids,
    _evidence_dto,
    _item,
    _message_dto,
    _structured_answer,
    _structured_run,
)
from llm_grounding.domain.answers import Answer
from llm_grounding.domain.corpus import (
    Chat,
    EvidenceReference,
    Message,
    RetrievalUnit,
)
from llm_grounding.domain.evidence import EvidenceType, EvidenceUnit
from llm_grounding.reasoning.basic_grounded import GroundedResponsePayload


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

    dto = _evidence_dto(chunk, {"evidence": evidence}, used=True)

    assert dto["chunk_id"] == "canonical-chunk"
    assert dto["reference_label"] == "paper_1"
    assert dto["used"] is True
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
    assert dto["has_sent_prompt"] is False


class EvidenceRepository:
    """Provide ordered retrieval chunks for browser DTO tests without a database."""

    def __init__(self, chunks: tuple[RetrievalUnit, ...]) -> None:
        """Index fixed chunks by their canonical identifiers."""
        self._chunks = {chunk.chunk_id: chunk for chunk in chunks}

    async def chunks_by_ids(
        self, chunk_ids: tuple[str, ...]
    ) -> dict[str, RetrievalUnit]:
        """Return requested chunks while preserving the DTO helper's supplied order."""
        return {
            chunk_id: self._chunks[chunk_id]
            for chunk_id in chunk_ids
            if chunk_id in self._chunks
        }

    async def evidence_by_ids(
        self, evidence_ids: tuple[str, ...]
    ) -> dict[str, EvidenceUnit]:
        """Return no page geometry for this focused response-shape test."""
        assert evidence_ids == ()
        return {}


class RunEvidenceRepository(EvidenceRepository):
    """Add fixed run metadata to the focused browser DTO repository."""

    def __init__(
        self,
        chunks: tuple[RetrievalUnit, ...],
        metadata: dict[str, object],
    ) -> None:
        """Retain one stored run alongside its canonical chunks."""
        super().__init__(chunks)
        self._metadata = metadata

    async def run_metadata(self, run_id: str) -> dict[str, object]:
        """Return the fixed run metadata for the requested test run."""
        assert run_id == "run"
        return self._metadata


class EvidenceApplication:
    """Expose only the repository dependency consumed by structured answer DTOs."""

    def __init__(self, chunks: tuple[RetrievalUnit, ...]) -> None:
        """Construct the minimal application shape used by `_structured_answer`."""
        self.repository = EvidenceRepository(chunks)


async def test_structured_answer_exposes_all_retrieved_chunks_with_used_state() -> None:
    """Canonical claims drive citations even if an answer's used-ID cache drifts."""
    cited = RetrievalUnit(
        "canonical-cited",
        "document",
        "ingestion",
        "Smith.pdf",
        "Supported text.",
        14,
        3,
        3,
        (),
    )
    additional = RetrievalUnit(
        "canonical-additional",
        "document",
        "ingestion",
        "Smith.pdf",
        "Additional context.",
        15,
        4,
        4,
        (),
    )
    payload = GroundedResponsePayload(
        turn_title="Smith evidence",
        answer_text="Supported text.",
        claims=[{"text": "Supported text.", "source_chunk_ids": [cited.chunk_id]}],
        insufficient_evidence=False,
    )
    answer = Answer(
        "Rendered answer",
        payload=payload.model_dump(),
        evidence_ids=(),
        metadata={
            "message_id": "message",
            "rag_run_id": "run",
            "turn_title": "Smith evidence",
            "retrieved_chunk_ids": [cited.chunk_id, additional.chunk_id],
        },
    )

    dto = await _structured_answer(
        EvidenceApplication((cited, additional)),  # type: ignore[arg-type]
        answer,
    )

    assert list(dto["evidence"]) == [cited.chunk_id, additional.chunk_id]
    assert dto["evidence"][cited.chunk_id]["reference_label"] == "Smith_15"
    assert dto["evidence"][cited.chunk_id]["used"] is True
    assert dto["evidence"][additional.chunk_id]["reference_label"] == "Smith_16"
    assert dto["evidence"][additional.chunk_id]["used"] is False


async def test_structured_answer_exposes_grounding_rejection() -> None:
    """A rejected answer remains visible with its decision reason and chunks."""
    answer = Answer(
        "Grounding failure: the generated response did not pass source validation.",
        metadata={
            "message_id": "message",
            "rag_run_id": "run",
            "grounding_rejected": True,
            "reason": "invalid_or_unvalidated_source_references",
            "retrieved_chunk_ids": [],
        },
    )

    dto = await _structured_answer(EvidenceApplication(()), answer)

    assert dto["status"] == "rejected"
    assert dto["reason"] == "invalid_or_unvalidated_source_references"
    assert dto["insufficient_evidence"] is False


def test_claim_source_chunk_ids_are_unique_in_first_citation_order() -> None:
    """The browser used subset follows canonical claims without duplicate IDs."""
    payload = GroundedResponsePayload(
        answer_text="Two supported claims.",
        claims=[
            {"text": "First.", "source_chunk_ids": ["canonical-b", "canonical-a"]},
            {"text": "Second.", "source_chunk_ids": ["canonical-a"]},
        ],
        insufficient_evidence=False,
    )

    assert _claim_source_chunk_ids(payload) == ("canonical-b", "canonical-a")


async def test_structured_run_derives_used_chunks_from_canonical_claims() -> None:
    """Restored answers ignore a stale presentation cache of evidence IDs."""
    cited = RetrievalUnit(
        "canonical-cited",
        "document",
        "ingestion",
        "Smith.pdf",
        "Supported text.",
        14,
        3,
        3,
        (),
    )
    additional = RetrievalUnit(
        "canonical-additional",
        "document",
        "ingestion",
        "Smith.pdf",
        "Additional context.",
        15,
        4,
        4,
        (),
    )
    payload = GroundedResponsePayload(
        turn_title="Smith evidence",
        answer_text="Supported text.",
        claims=[{"text": "Supported text.", "source_chunk_ids": [cited.chunk_id]}],
        insufficient_evidence=False,
    )
    repository = RunEvidenceRepository(
        (cited, additional),
        {
            "answer_payload": payload.model_dump(mode="json"),
            "evidence_ids": [additional.chunk_id],
            "retrieved_chunk_ids": [cited.chunk_id, additional.chunk_id],
        },
    )
    application = EvidenceApplication((cited, additional))
    application.repository = repository

    dto = await _structured_run(application, "run")  # type: ignore[arg-type]

    assert dto["claims"][0]["citations"] == [{"chunk_id": cited.chunk_id}]
    assert dto["evidence"][cited.chunk_id]["used"] is True
    assert dto["evidence"][additional.chunk_id]["used"] is False


async def test_structured_run_returns_abstention_for_empty_answer_payload() -> None:
    """A no-retrieval run remains renderable when no generated payload exists."""
    application = EvidenceApplication(())
    application.repository = RunEvidenceRepository(
        (),
        {
            "answer_payload": {},
            "answer_metadata": {"insufficient_evidence": True},
            "observation_status": "abstained",
            "retrieved_chunk_ids": [],
        },
    )

    dto = await _structured_run(application, "run")  # type: ignore[arg-type]

    assert dto["answer_text"].startswith("Insufficient evidence:")
    assert dto["status"] == "abstained"
    assert dto["insufficient_evidence"] is True
    assert dto["claims"] == []
    assert dto["evidence"] == {}


async def test_structured_run_exposes_persisted_grounding_rejection() -> None:
    """Restored rejected runs retain the rejection state for the chat UI."""
    application = EvidenceApplication(())
    application.repository = RunEvidenceRepository(
        (),
        {
            "answer_payload": {},
            "answer_metadata": {
                "grounding_rejected": True,
                "reason": "invalid_or_unvalidated_source_references",
            },
            "observation_status": "rejected",
            "retrieved_chunk_ids": [],
        },
    )

    dto = await _structured_run(application, "run")  # type: ignore[arg-type]

    assert dto["status"] == "rejected"
    assert dto["reason"] == "invalid_or_unvalidated_source_references"
    assert dto["insufficient_evidence"] is False


@pytest.mark.asyncio
async def test_missing_browser_resources_return_not_found_instead_of_server_error(
) -> None:
    """Browser resource lookups expose absent records as HTTP 404 responses."""

    class MissingRepository:
        async def run_metadata(self, run_id: str) -> dict[str, object]:
            raise KeyError(f"Unknown RAG run: {run_id}")

        async def get_document(self, document_id: str) -> object:
            raise KeyError(f"Unknown document: {document_id}")

    class MissingProjects:
        async def get_project(self, project_id: str) -> object:
            raise KeyError(f"Unknown project: {project_id}")

    class MissingChats:
        async def get_chat(self, chat_id: str) -> object:
            raise KeyError(f"Unknown chat: {chat_id}")

    from llm_grounding.api.main import create_api

    application = SimpleNamespace(
        evaluation=SimpleNamespace(corpus_service=None, embedding_store=None),
        repository=MissingRepository(),
        projects=MissingProjects(),
        chats=MissingChats(),
    )
    app = create_api(application)  # type: ignore[arg-type]

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        responses = await asyncio.gather(
            client.get("/api/runs/missing/evidence"),
            client.get("/api/projects/missing/chats"),
            client.get("/api/chats/missing/messages"),
            client.get("/api/documents/missing/pdf"),
        )

    assert [response.status_code for response in responses] == [404, 404, 404, 404]
