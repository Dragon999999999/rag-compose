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
from pydantic import BaseModel, Field, model_validator

from llm_grounding.api.eval import create_eval_router
from llm_grounding.api.eval_embeddings import create_eval_embeddings_router
from llm_grounding.application.bootstrap import (
    PhaseOneApplication,
    create_phase_one_application,
)
from llm_grounding.application.chat_service import PipelineSelectionError
from llm_grounding.application.eval_service import EvalService
from llm_grounding.domain.answers import Answer
from llm_grounding.domain.corpus import (
    Chat,
    Document,
    DocumentIngestion,
    DocumentSummary,
    IngestionStatus,
    Message,
    Pipeline,
    Project,
    RetrievalUnit,
    SummaryStatus,
    TaskType,
)
from llm_grounding.domain.evidence import EvidenceUnit
from llm_grounding.infrastructure.config.env import load_environment
from llm_grounding.infrastructure.config.loader import (
    load_architecture_config,
    load_ingestion_config,
)
from llm_grounding.infrastructure.config.models import PipelineConfig
from llm_grounding.infrastructure.config.settings import ApiSettings
from llm_grounding.reasoning.basic_grounded import GroundedResponsePayload
from llm_grounding.rendering.chunk_references import chunk_reference_label


class NameRequest(BaseModel):
    """Name a project or chat through the browser API."""

    name: str = Field(min_length=1, max_length=255)


class ChatCreateRequest(NameRequest):
    """Name a conversation and optionally select its initial pipeline."""

    pipeline_id: str | None = Field(default=None, min_length=1, max_length=36)


class ChatUpdateRequest(BaseModel):
    """Update one chat name or switch its persisted pipeline selection."""

    name: str | None = Field(default=None, min_length=1, max_length=255)
    pipeline_id: str | None = Field(default=None, min_length=1, max_length=36)

    @model_validator(mode="after")
    def require_update(self) -> "ChatUpdateRequest":
        """Reject empty PATCH requests while preserving explicit field semantics."""
        if self.name is None and self.pipeline_id is None:
            raise ValueError("At least one chat field must be provided")
        if self.name is not None and self.pipeline_id is not None:
            raise ValueError("Update chat name and pipeline selection separately")
        return self


class AskRequest(BaseModel):
    """Submit one persisted question or dedicated summary request."""

    message: str = Field(min_length=1, max_length=20_000)
    summary: bool = False
    pipeline_id: str | None = Field(default=None, min_length=1, max_length=36)


def _item(
    value: Project | Chat | Document | DocumentIngestion | DocumentSummary | Pipeline,
) -> dict[str, object]:
    """Serialize a small immutable domain record without framework leakage."""
    return cast(dict[str, object], asdict(value))


def _pipeline_http_error(error: PipelineSelectionError) -> HTTPException:
    """Translate a pipeline selection failure into a structured API envelope."""
    return HTTPException(
        error.status_code,
        detail={"code": error.code, "message": error.message},
    )


async def _source_dto(
    application: PhaseOneApplication,
    document: Document,
    ingestion: DocumentIngestion | None,
) -> dict[str, object]:
    """Serialize lightweight source-management data without loading its summary."""
    summary = (
        await application.repository.document_summary(document.id, ingestion.id)
        if ingestion is not None
        else None
    )
    return {
        "document": _item(document),
        "ingestion": _item(ingestion) if ingestion else None,
        "summary": _item(summary) if summary else None,
    }


async def _document_summary_dto(
    application: PhaseOneApplication,
    document_id: str,
    ingestion_id: str,
) -> dict[str, object]:
    """Return summary lifecycle state and its existing grounded provenance payload."""
    summary = await application.repository.document_summary(document_id, ingestion_id)
    if summary is None:
        return {
            "document_id": document_id,
            "ingestion_id": ingestion_id,
            "status": SummaryStatus.PENDING,
            "rag_run_id": None,
            "error": None,
            "page_count": None,
        }
    result = _item(summary)
    if summary.status == SummaryStatus.READY and summary.rag_run_id is not None:
        result["answer"] = await _structured_run(application, summary.rag_run_id)
    return result


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


def _claim_source_chunk_ids(
    payload: GroundedResponsePayload,
) -> tuple[str, ...]:
    """Collect canonical cited chunk IDs from the validated claim payload.

    The claim payload is the canonical grounding result. Deriving the used subset
    from it prevents a separately persisted presentation list from drifting away
    from the IDs used to construct browser citations.
    """
    return tuple(
        dict.fromkeys(
            chunk_id for claim in payload.claims for chunk_id in claim.source_chunk_ids
        )
    )


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
    payload = (
        GroundedResponsePayload.model_validate(answer.payload)
        if answer.payload
        else None
    )
    used_chunk_ids = (
        _claim_source_chunk_ids(payload) if payload else tuple(answer.evidence_ids)
    )
    retrieved_chunk_ids = _identifier_list(metadata.get("retrieved_chunk_ids"))
    visible_chunk_ids = tuple(dict.fromkeys((*retrieved_chunk_ids, *used_chunk_ids)))
    evidence = await _evidence_dtos(application, visible_chunk_ids, used_chunk_ids)
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
    if not isinstance(payload_data, dict):
        raise HTTPException(404, "Structured evidence is unavailable for this run.")
    payload = GroundedResponsePayload.model_validate(payload_data)
    used_chunk_ids = _claim_source_chunk_ids(payload)
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
        await application.startup()
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
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        allow_headers=[
            "Content-Type",
            "X-Request-ID",
            "Idempotency-Key",
            "X-Rag-Eval-Protocol",
            "X-Eval-Embedding-Token",
        ],
    )
    app.include_router(
        create_eval_router(
            EvalService(
                application, corpus_service=application.evaluation.corpus_service
            )
        )
    )
    app.include_router(
        create_eval_embeddings_router(application.evaluation.embedding_store)
    )

    @app.get("/api/pipelines")
    async def list_pipelines() -> list[dict[str, object]]:
        """List active pipeline metadata without returning full JSON documents."""
        if application.pipeline_service is None:
            raise HTTPException(503, "Pipeline management is unavailable.")
        records = await application.pipeline_service.list(active_only=True)
        return application.pipeline_service.summaries(records)

    @app.get("/api/pipelines/{pipeline_id}")
    async def get_pipeline(pipeline_id: str) -> dict[str, object]:
        """Return one complete persisted pipeline configuration."""
        if application.pipeline_service is None:
            raise HTTPException(503, "Pipeline management is unavailable.")
        try:
            return _item(await application.pipeline_service.get(pipeline_id))
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.post("/api/pipelines", status_code=201)
    async def create_pipeline(config: PipelineConfig) -> dict[str, object]:
        """Validate and persist an API-owned pipeline configuration."""
        if application.pipeline_service is None:
            raise HTTPException(503, "Pipeline management is unavailable.")
        try:
            return _item(await application.pipeline_service.create(config))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.put("/api/pipelines/{pipeline_id}")
    async def update_pipeline(
        pipeline_id: str, config: PipelineConfig
    ) -> dict[str, object]:
        """Replace a pipeline atomically and invalidate its runtime cache entry."""
        if application.pipeline_service is None:
            raise HTTPException(503, "Pipeline management is unavailable.")
        try:
            return _item(await application.pipeline_service.update(pipeline_id, config))
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.delete("/api/pipelines/{pipeline_id}", status_code=204)
    async def delete_pipeline(pipeline_id: str) -> None:
        """Deactivate a pipeline while retaining its persisted configuration."""
        if application.pipeline_service is None:
            raise HTTPException(503, "Pipeline management is unavailable.")
        try:
            await application.pipeline_service.deactivate(pipeline_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/api/catalog")
    async def catalog() -> dict[str, object]:
        """Expose the registered stage schemas and compatibility metadata."""
        if application.pipeline_service is None:
            raise HTTPException(503, "Pipeline management is unavailable.")
        return application.pipeline_service.catalog()

    @app.get("/api/models")
    async def models() -> dict[str, object]:
        """Expose credential-free embedding and reasoning model metadata."""
        if application.pipeline_service is None:
            raise HTTPException(503, "Pipeline management is unavailable.")
        return application.pipeline_service.models()

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
    async def create_chat(
        project_id: str, request: ChatCreateRequest
    ) -> dict[str, object]:
        try:
            chat = await application.chats.create_chat(
                project_id, request.name, request.pipeline_id
            )
        except PipelineSelectionError as exc:
            raise _pipeline_http_error(exc) from exc
        return _item(chat)

    @app.patch("/api/chats/{chat_id}")
    async def update_chat(
        chat_id: str, request: ChatUpdateRequest
    ) -> dict[str, object]:
        try:
            if request.pipeline_id is not None:
                chat = await application.chats.set_pipeline(
                    chat_id, request.pipeline_id
                )
            else:
                assert request.name is not None
                chat = await application.chats.rename_chat(chat_id, request.name)
        except PipelineSelectionError as exc:
            raise _pipeline_http_error(exc) from exc
        return _item(chat)

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
        try:
            answer = await application.chats.ask(
                chat_id,
                request.message,
                TaskType.SUMMARY if request.summary else TaskType.QA,
                request.pipeline_id,
            )
        except PipelineSelectionError as exc:
            raise _pipeline_http_error(exc) from exc
        return await _structured_answer(application, answer)

    @app.get("/api/runs/{run_id}/evidence")
    async def run_evidence(run_id: str) -> dict[str, object]:
        return await _structured_run(application, run_id)

    @app.get("/api/projects/{project_id}/documents")
    async def documents(project_id: str) -> list[dict[str, object]]:
        return [
            await _source_dto(application, document, ingestion)
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
            return await _source_dto(application, document, ingestion)

    @app.get("/api/projects/{project_id}/documents/{document_id}/summary")
    async def document_summary(project_id: str, document_id: str) -> dict[str, object]:
        """Load one project's existing grounded overview for an attached PDF."""
        source = next(
            (
                item
                for item in await application.projects.list_documents(project_id)
                if item[0].id == document_id and item[1] is not None
            ),
            None,
        )
        if source is None:
            raise HTTPException(404, "Document is not available in this project.")
        document, ingestion = source
        assert ingestion is not None
        if ingestion.status == IngestionStatus.READY:
            await application.summaries.schedule_summary(
                project_id, document, ingestion
            )
        return await _document_summary_dto(application, document.id, ingestion.id)

    @app.post("/api/projects/{project_id}/documents/{document_id}/summary/regenerate")
    async def regenerate_document_summary(
        project_id: str, document_id: str
    ) -> dict[str, object]:
        """Explicitly regenerate a ready PDF's grounded document overview."""
        source = next(
            (
                item
                for item in await application.projects.list_documents(project_id)
                if item[0].id == document_id and item[1] is not None
            ),
            None,
        )
        if source is None:
            raise HTTPException(404, "Document is not available in this project.")
        document, ingestion = source
        assert ingestion is not None
        if ingestion.status != IngestionStatus.READY:
            raise HTTPException(
                409, "Document ingestion is not ready for summarization."
            )
        await application.summaries.regenerate_summary(project_id, document, ingestion)
        return await _document_summary_dto(application, document.id, ingestion.id)

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
    load_environment()
    architecture = load_architecture_config(
        Path("configs/architectures/basic_rag_v1.yaml")
    )
    application = create_phase_one_application(
        architecture, load_ingestion_config(None)
    )
    api_settings = ApiSettings.from_environment()
    uvicorn.run(
        create_api(application), host=api_settings.host, port=api_settings.port
    )
