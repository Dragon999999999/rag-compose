"""FastAPI adapter exposing application services and validated provenance DTOs."""

import os
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import cast

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from llm_grounding.application.bootstrap import (
    PhaseOneApplication,
    create_phase_one_application,
)
from llm_grounding.domain.answers import Answer
from llm_grounding.domain.corpus import (
    Chat,
    Document,
    DocumentIngestion,
    Message,
    Project,
    RetrievalUnit,
    TaskType,
)
from llm_grounding.domain.evidence import EvidenceUnit
from llm_grounding.infrastructure.config.loader import (
    load_architecture_config,
    load_ingestion_config,
)
from llm_grounding.reasoning.basic_grounded import GroundedResponsePayload
from llm_grounding.rendering.chunk_references import chunk_reference_label


class NameRequest(BaseModel):
    """Name a project or chat through the browser API."""

    name: str = Field(min_length=1, max_length=255)


class AskRequest(BaseModel):
    """Submit one persisted question or dedicated summary request."""

    message: str = Field(min_length=1, max_length=20_000)
    summary: bool = False


def _item(
    value: Project | Chat | Document | DocumentIngestion,
) -> dict[str, object]:
    """Serialize a small immutable domain record without framework leakage."""
    return cast(dict[str, object], asdict(value))


def _message_dto(message: Message) -> dict[str, object]:
    """Return persisted chat text and a separately resolved provenance payload."""
    return {
        "id": message.id,
        "chat_id": message.chat_id,
        "role": message.role,
        "content": message.content,
        "created_at": message.created_at,
        "rag_run_id": message.rag_run_id,
        "turn_title": message.turn_title,
    }


def _evidence_dto(
    chunk: RetrievalUnit,
    evidence: dict[str, EvidenceUnit],
    used: bool,
) -> dict[str, object]:
    """Expose a trusted chunk with stable display label and PDF locations."""
    references = chunk.evidence_refs
    locations: list[dict[str, object]] = []
    for reference in references:
        item = evidence.get(reference.evidence_id)
        if item is None or item.page_number is None:
            continue
        box = item.bounding_box
        locations.append(
            {
                "page": item.page_number,
                "bounding_boxes": (
                    [
                        {
                            "x": box[0],
                            "y": box[1],
                            "width": box[2] - box[0],
                            "height": box[3] - box[1],
                        }
                    ]
                    if box is not None
                    else []
                ),
                "positioning": "exact" if box is not None else "page_only",
            }
        )
    return {
        "chunk_id": chunk.chunk_id,
        "reference_label": chunk_reference_label(chunk),
        "used": used,
        "document_id": chunk.document_id,
        "document_name": chunk.document_name,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "excerpt": chunk.text,
        "locations": locations,
    }


def _identifier_list(value: object) -> tuple[str, ...]:
    """Return unique string identifiers while preserving their supplied order."""
    if not isinstance(value, list):
        return ()
    return tuple(dict.fromkeys(item for item in value if isinstance(item, str)))


async def _evidence_dtos(
    application: PhaseOneApplication,
    chunk_ids: tuple[str, ...],
    used_chunk_ids: tuple[str, ...],
) -> dict[str, dict[str, object]]:
    """Hydrate ordered retrieved chunks and mark the subset cited by claims."""
    chunks_by_id = await application.repository.chunks_by_ids(chunk_ids)
    chunks = tuple(
        chunks_by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in chunks_by_id
    )
    evidence_ids = tuple(
        reference.evidence_id for chunk in chunks for reference in chunk.evidence_refs
    )
    evidence = await application.repository.evidence_by_ids(evidence_ids)
    used = set(used_chunk_ids)
    return {
        chunk.chunk_id: _evidence_dto(chunk, evidence, chunk.chunk_id in used)
        for chunk in chunks
    }


async def _structured_answer(
    application: PhaseOneApplication,
    answer: Answer,
) -> dict[str, object]:
    """Build browser-safe claims and evidence from canonical validated output."""
    metadata = answer.metadata
    used_chunk_ids = tuple(answer.evidence_ids)
    retrieved_chunk_ids = _identifier_list(metadata.get("retrieved_chunk_ids"))
    visible_chunk_ids = tuple(dict.fromkeys((*retrieved_chunk_ids, *used_chunk_ids)))
    evidence = await _evidence_dtos(application, visible_chunk_ids, used_chunk_ids)
    payload = (
        GroundedResponsePayload.model_validate(answer.payload)
        if answer.payload
        else None
    )
    return {
        "message_id": metadata["message_id"],
        "rag_run_id": metadata["rag_run_id"],
        "turn_title": metadata.get("turn_title"),
        "answer_text": payload.answer_text if payload else answer.text,
        "insufficient_evidence": (
            payload.insufficient_evidence
            if payload
            else bool(answer.metadata.get("insufficient_evidence", False))
        ),
        "claims": [
            {
                "id": f"claim_{index}",
                "text": claim.text,
                "citations": [
                    {"chunk_id": chunk_id}
                    for chunk_id in claim.source_chunk_ids
                    if chunk_id in evidence
                ],
            }
            for index, claim in enumerate(payload.claims if payload else (), start=1)
        ],
        "evidence": {chunk_id: item for chunk_id, item in evidence.items()},
        "retrieved_chunk_ids": list(retrieved_chunk_ids),
    }


async def _structured_run(
    application: PhaseOneApplication, run_id: str
) -> dict[str, object]:
    """Restore persisted structured provenance for an older assistant message."""
    metadata = await application.repository.run_metadata(run_id)
    payload_data = metadata.get("answer_payload", {})
    evidence_data = metadata.get("evidence_ids", [])
    if not isinstance(payload_data, dict) or not isinstance(evidence_data, list):
        raise HTTPException(404, "Structured evidence is unavailable for this run.")
    payload = GroundedResponsePayload.model_validate(payload_data)
    used_chunk_ids = _identifier_list(evidence_data)
    retrieved_chunk_ids = _identifier_list(metadata.get("retrieved_chunk_ids"))
    visible_chunk_ids = tuple(dict.fromkeys((*retrieved_chunk_ids, *used_chunk_ids)))
    evidence = await _evidence_dtos(
        application,
        visible_chunk_ids,
        used_chunk_ids,
    )
    return {
        "rag_run_id": run_id,
        "turn_title": payload.turn_title,
        "answer_text": payload.answer_text,
        "insufficient_evidence": payload.insufficient_evidence,
        "claims": [
            {
                "id": f"claim_{index}",
                "text": claim.text,
                "citations": [
                    {"chunk_id": chunk_id}
                    for chunk_id in claim.source_chunk_ids
                    if chunk_id in evidence
                ],
            }
            for index, claim in enumerate(payload.claims, start=1)
        ],
        "evidence": {chunk_id: item for chunk_id, item in evidence.items()},
        "retrieved_chunk_ids": list(retrieved_chunk_ids),
    }


def create_api(application: PhaseOneApplication) -> FastAPI:
    """Create the HTTP adapter without moving business logic into route handlers."""

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        await application.close()

    app = FastAPI(title="LLM Grounding API", lifespan=lifespan)
    origins = [
        origin
        for origin in os.getenv(
            "LLM_GROUNDING_CORS_ORIGINS", "http://localhost:5173"
        ).split(",")
        if origin
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        allow_headers=["Content-Type"],
    )

    @app.get("/api/projects")
    async def list_projects() -> list[dict[str, object]]:
        return [
            _item(project) for project in await application.projects.list_projects()
        ]

    @app.post("/api/projects")
    async def create_project(request: NameRequest) -> dict[str, object]:
        return _item(await application.projects.create_project(request.name))

    @app.patch("/api/projects/{project_id}")
    async def rename_project(
        project_id: str, request: NameRequest
    ) -> dict[str, object]:
        return _item(
            await application.projects.rename_project(project_id, request.name)
        )

    @app.delete("/api/projects/{project_id}", status_code=204)
    async def delete_project(project_id: str) -> None:
        await application.projects.delete_project(project_id)

    @app.get("/api/projects/{project_id}/chats")
    async def list_chats(project_id: str) -> list[dict[str, object]]:
        return [_item(chat) for chat in await application.chats.list_chats(project_id)]

    @app.post("/api/projects/{project_id}/chats")
    async def create_chat(project_id: str, request: NameRequest) -> dict[str, object]:
        return _item(await application.chats.create_chat(project_id, request.name))

    @app.patch("/api/chats/{chat_id}")
    async def rename_chat(chat_id: str, request: NameRequest) -> dict[str, object]:
        return _item(await application.chats.rename_chat(chat_id, request.name))

    @app.delete("/api/chats/{chat_id}", status_code=204)
    async def delete_chat(chat_id: str) -> None:
        await application.chats.delete_chat(chat_id)

    @app.get("/api/chats/{chat_id}/messages")
    async def messages(chat_id: str) -> list[dict[str, object]]:
        return [
            _message_dto(message)
            for message in await application.chats.messages(chat_id)
        ]

    @app.post("/api/chats/{chat_id}/messages")
    async def ask(chat_id: str, request: AskRequest) -> dict[str, object]:
        answer = await application.chats.ask(
            chat_id,
            request.message,
            TaskType.SUMMARY if request.summary else TaskType.QA,
        )
        return await _structured_answer(application, answer)

    @app.get("/api/runs/{run_id}/evidence")
    async def run_evidence(run_id: str) -> dict[str, object]:
        return await _structured_run(application, run_id)

    @app.get("/api/projects/{project_id}/documents")
    async def documents(project_id: str) -> list[dict[str, object]]:
        return [
            {
                "document": _item(document),
                "ingestion": _item(ingestion) if ingestion else None,
            }
            for document, ingestion in await application.projects.list_documents(
                project_id
            )
        ]

    @app.post("/api/projects/{project_id}/documents")
    async def upload_document(
        project_id: str,
        file: UploadFile = File(...),  # noqa: B008
    ) -> dict[str, object]:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(422, "Only PDF uploads are supported.")
        original_name = Path(file.filename).name
        with tempfile.TemporaryDirectory(prefix="llm-grounding-upload-") as directory:
            temporary_path = Path(directory) / original_name
            temporary_path.write_bytes(await file.read())
            document, ingestion = await application.projects.add_document(
                project_id, temporary_path, load_ingestion_config(None)
            )
            return {"document": _item(document), "ingestion": _item(ingestion)}

    @app.delete("/api/projects/{project_id}/documents/{document_id}", status_code=204)
    async def remove_document(project_id: str, document_id: str) -> None:
        await application.projects.remove_document(project_id, document_id)

    @app.get("/api/documents/{document_id}/pdf")
    async def pdf(document_id: str) -> FileResponse:
        document = await application.repository.get_document(document_id)
        path = await application.artifacts.path_for(document.artifact_key)
        return FileResponse(path, media_type="application/pdf", filename=document.name)

    return app


def main() -> None:
    """Start the browser API using the existing Basic RAG composition root."""
    architecture = load_architecture_config(
        Path("configs/architectures/basic_rag_v1.yaml")
    )
    application = create_phase_one_application(
        architecture, load_ingestion_config(None)
    )
    uvicorn.run(create_api(application), host="127.0.0.1", port=8000)
