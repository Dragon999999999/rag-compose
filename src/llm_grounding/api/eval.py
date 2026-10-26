"""Native versioned HTTP adapter for reproducible RAG evaluation."""

from typing import Any

from fastapi import APIRouter, Header
from fastapi.responses import JSONResponse

from llm_grounding.api.eval_schemas import (
    EvalConfigOverrides,
    EvalCorpusRequest,
    EvalError,
    EvalInventoryUpload,
    EvalQueryRequest,
    EvalRetrieveRequest,
)
from llm_grounding.application.eval_service import EvalService, EvalServiceError
from llm_grounding.observability.models import EffectiveConfiguration


def create_eval_router(service: EvalService) -> APIRouter:
    """Create the ``/eval/v1`` router around the application evaluation service."""
    router = APIRouter(prefix="/eval/v1", tags=["evaluation"])

    @router.get("/health")
    async def health() -> dict[str, Any]:
        """Report that the evaluation adapter and its configured dependencies exist."""
        application = service._application  # noqa: SLF001
        return {
            "status": "ok",
            "protocol_version": "1",
            "dependencies": {
                "repository": (
                    "configured" if application.repository else "unavailable"
                ),
                "vectors": (
                    "configured"
                    if getattr(application, "_vectors", None)
                    else "unavailable"
                ),
            },
        }

    @router.get("/capabilities")
    async def capabilities() -> dict[str, Any]:
        """Advertise implemented protocol features only."""
        return {
            "protocol_version": "1",
            "features": [
                "query",
                "retrieve",
                "rerank",
                "observations",
                "usage",
                "effective_config",
            ],
            "unsupported": [
                "claim_entailment",
                "judging",
                "fusion",
                "filter",
                "compression",
                "confidence_calibration",
                "multi_hop_decomposition",
                "provider_seed",
            ],
        }

    @router.get("/config-schema")
    async def config_schema() -> dict[str, Any]:
        """Expose supported request overrides and the recorded effective schema."""
        return {
            "protocol_version": "1",
            "supported_pipeline": "query_basic_v1",
            "schema": {
                "overrides": EvalConfigOverrides.model_json_schema(),
                "effective_configuration": EffectiveConfiguration.model_json_schema(),
            },
        }

    @router.post("/corpora")
    async def create_corpus(body: EvalCorpusRequest) -> dict[str, Any]:
        """Create an isolated benchmark corpus inventory."""
        corpus = await service.create_corpus(**body.model_dump())
        return corpus.model_dump(mode="json")

    @router.get("/corpora/{corpus_id}", response_model=None)
    async def get_corpus(corpus_id: str) -> dict[str, Any] | JSONResponse:
        """Return an isolated benchmark corpus inventory."""
        try:
            corpus = await service.get_corpus(corpus_id)
        except KeyError as exc:
            return _error_response(EvalServiceError("corpus_not_found", str(exc), 404))
        return corpus.model_dump(mode="json")

    @router.delete("/corpora/{corpus_id}", status_code=204, response_model=None)
    async def delete_corpus(corpus_id: str) -> None | JSONResponse:
        """Delete only the isolated benchmark namespace."""
        try:
            await service.delete_corpus(corpus_id)
        except EvalServiceError as exc:
            return _error_response(exc)
        return None

    @router.post("/corpora/{corpus_id}/documents", response_model=None)
    async def upload_documents(
        corpus_id: str, body: EvalInventoryUpload
    ) -> dict[str, Any] | JSONResponse:
        """Append source-document inventory and return its operation identity."""
        try:
            return await service.upload_corpus_documents(corpus_id, body.items)
        except EvalServiceError as exc:
            return _error_response(exc)

    @router.post("/corpora/{corpus_id}/chunks", response_model=None)
    async def upload_chunks(
        corpus_id: str, body: EvalInventoryUpload
    ) -> dict[str, Any] | JSONResponse:
        """Append canonical chunks to the isolated benchmark inventory."""
        try:
            return await service.upload_corpus_chunks(corpus_id, body.items)
        except EvalServiceError as exc:
            return _error_response(exc)

    @router.get("/corpora/{corpus_id}/documents", response_model=None)
    async def corpus_documents(corpus_id: str) -> dict[str, Any] | JSONResponse:
        """Read isolated source-document inventory."""
        return await _corpus_inventory(service, corpus_id, "documents")

    @router.get("/corpora/{corpus_id}/chunks", response_model=None)
    async def corpus_chunks(corpus_id: str) -> dict[str, Any] | JSONResponse:
        """Read isolated canonical chunk inventory."""
        return await _corpus_inventory(service, corpus_id, "chunks")

    @router.get("/corpora/{corpus_id}/evidence", response_model=None)
    async def corpus_evidence(corpus_id: str) -> dict[str, Any] | JSONResponse:
        """Read evidence/source spans supplied with the isolated chunk inventory."""
        try:
            corpus = await service.get_corpus(corpus_id)
        except EvalServiceError as exc:
            return _error_response(exc)
        evidence = [
            item
            for chunk in corpus.chunks
            for item in chunk.get("evidence", chunk.get("source_spans", []))
            if isinstance(item, dict)
        ]
        return {"corpus_id": corpus_id, "evidence": evidence}

    @router.get("/operations/{operation_id}", response_model=None)
    async def operation(operation_id: str) -> dict[str, Any] | JSONResponse:
        """Return isolated ingestion/upload operation status."""
        try:
            return await service.operation(operation_id)
        except EvalServiceError as exc:
            return _error_response(exc)

    @router.post("/query", response_model=None)
    async def query(
        body: EvalQueryRequest,
        x_request_id: str | None = Header(default=None),
        idempotency_key: str | None = Header(default=None),
        x_rag_eval_protocol: str | None = Header(default=None),
    ) -> dict[str, Any]:
        """Run one full query and return the canonical observation."""
        try:
            _validate_protocol(x_rag_eval_protocol)
            observation = await service.query(
                body, request_id=x_request_id, idempotency_key=idempotency_key
            )
        except EvalServiceError as exc:
            return _error_response(exc)
        return observation.model_dump(mode="json", exclude_none=True)

    @router.post("/retrieve", response_model=None)
    async def retrieve(
        body: EvalRetrieveRequest,
        x_request_id: str | None = Header(default=None),
        idempotency_key: str | None = Header(default=None),
        x_rag_eval_protocol: str | None = Header(default=None),
    ) -> dict[str, Any]:
        """Run retrieval and final-context construction without generation."""
        try:
            _validate_protocol(x_rag_eval_protocol)
            observation = await service.retrieve(
                body, request_id=x_request_id, idempotency_key=idempotency_key
            )
        except EvalServiceError as exc:
            return _error_response(exc)
        return observation.model_dump(mode="json", exclude_none=True)

    @router.get("/requests/{request_id}", response_model=None)
    async def request_observation(
        request_id: str,
        x_rag_eval_protocol: str | None = Header(default=None),
    ) -> dict[str, Any]:
        """Restore one persisted observation by protocol request identity."""
        try:
            _validate_protocol(x_rag_eval_protocol)
            observation = await service.by_request_id(request_id)
        except EvalServiceError as exc:
            return _error_response(exc)
        return observation.model_dump(mode="json", exclude_none=True)

    return router


def _validate_protocol(value: str | None) -> None:
    """Reject an explicitly unsupported protocol version."""
    if value is not None and value != "1":
        raise EvalServiceError(
            "unsupported_protocol", "X-Rag-Eval-Protocol must be '1'.", 400
        )


async def _corpus_inventory(
    service: EvalService, corpus_id: str, key: str
) -> dict[str, Any] | JSONResponse:
    """Serialize one corpus inventory collection with a stable envelope."""
    try:
        corpus = await service.get_corpus(corpus_id)
    except EvalServiceError as exc:
        return _error_response(exc)
    return {"corpus_id": corpus_id, key: getattr(corpus, key)}


def _error_response(exc: EvalServiceError) -> JSONResponse:
    """Return a protocol error without exposing exception internals."""
    payload = EvalError(code=exc.code, message=exc.message).model_dump(mode="json")
    return JSONResponse(payload, status_code=exc.status_code)
