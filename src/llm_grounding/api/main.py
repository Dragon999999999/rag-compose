"""FastAPI adapter exposing application services and validated provenance DTOs."""

import os
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import cast

import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, model_validator

from llm_grounding.api.eval import create_eval_router
from llm_grounding.api.eval_embeddings import (
    EmbeddingFetcher,
    create_eval_embeddings_router,
)
from llm_grounding.application.bootstrap import (
    PhaseOneApplication,
    create_phase_one_application,
)
from llm_grounding.application.chat_service import PipelineSelectionError
from llm_grounding.application.eval_service import EvalService
from llm_grounding.application.pipeline_service import (
    PipelineCompatibilityError,
    PipelineConcurrencyError,
    PipelineDependencyError,
)
from llm_grounding.domain.answers import Answer
from llm_grounding.domain.corpus import (
    Chat,
    Document,
    DocumentIngestion,
    Message,
    Pipeline,
    Project,
    RetrievalUnit,
    TaskType,
)
from llm_grounding.domain.evidence import EvidenceUnit
from llm_grounding.infrastructure.config.env import load_environment
from llm_grounding.infrastructure.config.loader import (
    load_ingestion_config,
    load_pipeline_config,
)
from llm_grounding.infrastructure.config.models import PipelineConfig
from llm_grounding.infrastructure.config.settings import ApiSettings
from llm_grounding.reasoning.basic_grounded import GroundedResponsePayload
from llm_grounding.rendering.chunk_references import chunk_reference_label


class NameRequest(BaseModel):
    """Name a project or chat through the browser API."""

    name: str = Field(min_length=1, max_length=255)


class ProjectCreateRequest(NameRequest):
    """Create a project and optionally select its ingestion pipeline."""

    ingestion_pipeline_id: str | None = Field(default=None, min_length=1, max_length=36)


class ProjectUpdateRequest(BaseModel):
    """Rename a project or change its ingestion pipeline selection."""

    name: str | None = Field(default=None, min_length=1, max_length=255)
    ingestion_pipeline_id: str | None = Field(default=None, min_length=1, max_length=36)

    @model_validator(mode="after")
    def require_update(self) -> "ProjectUpdateRequest":
        """Require exactly one project mutation per request."""
        if self.name is None and self.ingestion_pipeline_id is None:
            raise ValueError("At least one project field must be provided")
        if self.name is not None and self.ingestion_pipeline_id is not None:
            raise ValueError("Update project name and ingestion selection separately")
        return self


class ChatCreateRequest(NameRequest):
    """Name a conversation and optionally select its initial pipeline."""

    pipeline_id: str | None = Field(default=None, min_length=1, max_length=36)
    stage_overrides: dict[str, dict[str, object]] = Field(default_factory=dict)


class ChatUpdateRequest(BaseModel):
    """Update one chat name or switch its persisted pipeline selection."""

    name: str | None = Field(default=None, min_length=1, max_length=255)
    pipeline_id: str | None = Field(default=None, min_length=1, max_length=36)
    stage_overrides: dict[str, dict[str, object]] | None = None

    @model_validator(mode="after")
    def require_update(self) -> "ChatUpdateRequest":
        """Reject empty PATCH requests while preserving explicit field semantics."""
        values = (self.name, self.pipeline_id, self.stage_overrides)
        if all(value is None for value in values):
            raise ValueError("At least one chat field must be provided")
        if sum(value is not None for value in values) > 1:
            raise ValueError("Update one chat field at a time")
        return self


class AskRequest(BaseModel):
    """Submit one persisted question to a chat."""

    message: str = Field(min_length=1, max_length=20_000)
    pipeline_id: str | None = Field(default=None, min_length=1, max_length=36)
    stage_overrides: dict[str, dict[str, object]] = Field(default_factory=dict)


def _item(
    value: Project | Chat | Document | DocumentIngestion | Pipeline,
) -> dict[str, object]:
    """Serialize a small immutable domain record without framework leakage."""
    item = cast(dict[str, object], asdict(value))
    if isinstance(value, Pipeline):
        config = item.get("config")
        if isinstance(config, dict):
            # Resolved provider snapshots are execution metadata, not editable
            # pipeline source and must not be echoed as API configuration fields.
            if config.get("resolved_catalog") is not None:
                item["config"] = {
                    **config,
                    "resolved_catalog": None,
                }
    return item


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
    """Serialize lightweight source-management data."""
    return {
        "document": _item(document),
        "ingestion": _item(ingestion) if ingestion else None,
    }


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
    retrieved_chunk_ids = _identifier_list(metadata.get("retrieved_chunk_ids"))
    if not isinstance(payload_data, dict) or not payload_data:
        answer_metadata = metadata.get("answer_metadata")
        answer_metadata = (
            answer_metadata if isinstance(answer_metadata, dict) else {}
        )
        rejected = (
            metadata.get("status") == "rejected"
            or metadata.get("observation_status") == "rejected"
            or answer_metadata.get("grounding_rejected") is True
        )
        evidence = await _evidence_dtos(
            application,
            retrieved_chunk_ids,
            (),
        )
        return {
            "rag_run_id": run_id,
            "turn_title": "Earlier answer",
            "answer_text": (
                "Grounding failure: the generated response did not pass source "
                "validation."
                if rejected
                else "Insufficient evidence: no indexed project chunks matched "
                "this request."
            ),
            "insufficient_evidence": not rejected,
            "claims": [],
            "evidence": {chunk_id: item for chunk_id, item in evidence.items()},
            "retrieved_chunk_ids": list(retrieved_chunk_ids),
        }
    payload = GroundedResponsePayload.model_validate(payload_data)
    used_chunk_ids = _claim_source_chunk_ids(payload)
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
    if application.evaluation.embedding_store is not None and hasattr(
        application.evaluation.embedding_store, "fetch_vectors"
    ):
        app.include_router(
            create_eval_embeddings_router(
                cast(EmbeddingFetcher, application.evaluation.embedding_store)
            )
        )

    @app.get("/api/pipelines")
    async def list_pipelines(
        include_inactive: bool = Query(False),
        kind: str | None = Query(None),
    ) -> list[dict[str, object]]:
        """List pipeline metadata without returning full JSON documents."""
        if application.pipeline_service is None:
            raise HTTPException(503, "Pipeline management is unavailable.")
        try:
            records = await application.pipeline_service.list(
                active_only=not include_inactive, kind=kind
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
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

    @app.post("/api/pipelines/{pipeline_id}/impact")
    async def pipeline_update_impact(
        pipeline_id: str, config: PipelineConfig
    ) -> dict[str, object]:
        """Return authoritative dependency and compatibility impact for a draft."""
        if application.pipeline_service is None:
            raise HTTPException(503, "Pipeline management is unavailable.")
        try:
            impact = await application.pipeline_service.impact(pipeline_id, config)
            return impact.as_dict()
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.put("/api/pipelines/{pipeline_id}")
    async def update_pipeline(
        pipeline_id: str,
        config: PipelineConfig,
        expected_config_hash: str | None = Query(None),
        migrate_conversations: bool = Query(False),
    ) -> dict[str, object]:
        """Update a pipeline with explicit migration and stale-draft checks."""
        if application.pipeline_service is None:
            raise HTTPException(503, "Pipeline management is unavailable.")
        try:
            return _item(
                await application.pipeline_service.update(
                    pipeline_id,
                    config,
                    expected_config_hash=expected_config_hash,
                    migrate_conversations=migrate_conversations,
                )
            )
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except PipelineDependencyError as exc:
            raise HTTPException(
                409,
                detail={
                    "code": "pipeline_dependencies",
                    "message": str(exc),
                    "impact": exc.impact.as_dict(),
                },
            ) from exc
        except PipelineCompatibilityError as exc:
            raise HTTPException(
                409,
                detail={
                    "code": "incompatible_representation",
                    "message": str(exc),
                    "impact": exc.impact.as_dict(),
                },
            ) from exc
        except PipelineConcurrencyError as exc:
            raise HTTPException(
                409,
                detail={"code": "pipeline_stale", "message": str(exc)},
            ) from exc
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
    async def create_project(request: ProjectCreateRequest) -> dict[str, object]:
        try:
            return _item(
                await application.projects.create_project(
                    request.name, request.ingestion_pipeline_id
                )
            )
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.patch("/api/projects/{project_id}")
    async def update_project(
        project_id: str, request: ProjectUpdateRequest
    ) -> dict[str, object]:
        try:
            if request.ingestion_pipeline_id is not None:
                project = await application.projects.set_ingestion_pipeline(
                    project_id, request.ingestion_pipeline_id
                )
            else:
                assert request.name is not None
                project = await application.projects.rename_project(
                    project_id, request.name
                )
            return _item(project)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(
                409,
                detail={"code": "incompatible_ingestion", "message": str(exc)},
            ) from exc

    @app.delete("/api/projects/{project_id}", status_code=204)
    async def delete_project(project_id: str) -> None:
        try:
            await application.projects.delete_project(project_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/api/projects/{project_id}/compatible-query-pipelines")
    async def compatible_query_pipelines(
        project_id: str,
    ) -> list[dict[str, object]]:
        """Return query pipelines and their compatibility with a project index."""
        try:
            return await application.projects.compatible_query_pipelines(project_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.get("/api/projects/{project_id}/chats")
    async def list_chats(project_id: str) -> list[dict[str, object]]:
        try:
            project_service = getattr(application, "projects", None)
            if project_service is not None:
                await project_service.get_project(project_id)
            chats = await application.chats.list_chats(project_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        return [_item(chat) for chat in chats]

    @app.post("/api/projects/{project_id}/chats")
    async def create_chat(
        project_id: str, request: ChatCreateRequest
    ) -> dict[str, object]:
        try:
            chat = await application.chats.create_chat(
                project_id,
                request.name,
                request.pipeline_id,
                stage_overrides=request.stage_overrides,
            )
        except PipelineSelectionError as exc:
            raise _pipeline_http_error(exc) from exc
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
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
            elif request.stage_overrides is not None:
                chat = await application.chats.set_stage_overrides(
                    chat_id, request.stage_overrides
                )
            else:
                assert request.name is not None
                chat = await application.chats.rename_chat(chat_id, request.name)
        except PipelineSelectionError as exc:
            raise _pipeline_http_error(exc) from exc
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        return _item(chat)

    @app.delete("/api/chats/{chat_id}", status_code=204)
    async def delete_chat(chat_id: str) -> None:
        try:
            await application.chats.delete_chat(chat_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/api/chats/{chat_id}/messages")
    async def messages(chat_id: str) -> list[dict[str, object]]:
        try:
            await application.chats.get_chat(chat_id)
            chat_messages = await application.chats.messages(chat_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        return [_message_dto(message) for message in chat_messages]

    @app.post("/api/chats/{chat_id}/messages")
    async def ask(chat_id: str, request: AskRequest) -> dict[str, object]:
        try:
            answer = await application.chats.ask(
                chat_id,
                request.message,
                TaskType.QA,
                request.pipeline_id,
                request.stage_overrides,
            )
        except PipelineSelectionError as exc:
            raise _pipeline_http_error(exc) from exc
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        return await _structured_answer(application, answer)

    @app.get("/api/runs/{run_id}/evidence")
    async def run_evidence(run_id: str) -> dict[str, object]:
        try:
            return await _structured_run(application, run_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/api/projects/{project_id}/documents")
    async def documents(project_id: str) -> list[dict[str, object]]:
        try:
            await application.projects.get_project(project_id)
            sources = await application.projects.list_documents(project_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        return [
            await _source_dto(application, document, ingestion)
            for document, ingestion in sources
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
            try:
                document, ingestion = await application.projects.add_document(
                    project_id, temporary_path, load_ingestion_config(None)
                )
            except KeyError as exc:
                raise HTTPException(404, str(exc)) from exc
            return await _source_dto(application, document, ingestion)

    @app.delete("/api/projects/{project_id}/documents/{document_id}", status_code=204)
    async def remove_document(project_id: str, document_id: str) -> None:
        try:
            await application.projects.remove_document(project_id, document_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/api/documents/{document_id}/pdf")
    async def pdf(document_id: str) -> FileResponse:
        try:
            document = await application.repository.get_document(document_id)
            path = await application.artifacts.path_for(document.artifact_key)
        except (KeyError, FileNotFoundError) as exc:
            raise HTTPException(404, str(exc)) from exc
        return FileResponse(path, media_type="application/pdf", filename=document.name)

    return app


def main() -> None:
    """Start the browser API using the existing Basic RAG composition root."""
    load_environment()
    architecture = load_pipeline_config(Path("config/pipelines/query_basic_v1.yaml"))
    application = create_phase_one_application(
        architecture, load_ingestion_config(None)
    )
    api_settings = ApiSettings.from_environment()
    uvicorn.run(
        create_api(application), host=api_settings.host, port=api_settings.port
    )
