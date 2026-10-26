"""Application service for the native ``/eval/v1`` protocol."""

import json
from datetime import UTC, datetime
from hashlib import sha256
from uuid import uuid4

from llm_grounding.api.eval_schemas import EvalQueryRequest, EvalRetrieveRequest
from llm_grounding.application.bootstrap import PhaseOneApplication
from llm_grounding.application.eval_config import EvaluationConfigService
from llm_grounding.application.eval_corpus_service import EvalCorpusService
from llm_grounding.domain.corpus import ChunkRetrievalResult, CorpusRequest, Message
from llm_grounding.observability.builder import ObservationBuilder
from llm_grounding.observability.configuration import build_effective_configuration
from llm_grounding.observability.models import RagRunObservation


class EvalServiceError(ValueError):
    """Report a protocol error with an HTTP-appropriate status and stable code."""

    def __init__(self, code: str, message: str, status_code: int = 400) -> None:
        """Retain safe client-facing error data."""
        super().__init__(message)
        self.code, self.message, self.status_code = code, message, status_code


class EvalService:
    """Orchestrate evaluation requests through the existing Basic RAG runner."""

    def __init__(
        self,
        application: PhaseOneApplication,
        builder: ObservationBuilder | None = None,
        corpus_service: EvalCorpusService | None = None,
    ) -> None:
        """Inject the application composition and canonical observation builder."""
        self._application = application
        self._builder = builder or ObservationBuilder()
        self._corpora = corpus_service or EvalCorpusService(application.repository)
        self._config = EvaluationConfigService()

    async def query(
        self,
        request: EvalQueryRequest,
        *,
        request_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> RagRunObservation:
        """Run generation and persist the canonical observation."""
        fingerprint = self._fingerprint("query", request)
        identifier = (
            request_id or idempotency_key or self._deterministic_id(fingerprint)
        )
        existing = await self._existing(identifier, fingerprint)
        if existing is not None:
            return existing
        runner = self._runner()
        overrides = self._overrides(request)
        effective = self._effective(request.corpus_id, overrides)
        corpus_request = CorpusRequest(
            request.corpus_id,
            request.query,
            document_ids=self._document_ids(request.filters),
        )
        history = self._history(request.history)
        try:
            if overrides:
                run_id, answer, metadata = await runner.run(
                    corpus_request,
                    history,
                    runtime_config=overrides,
                )
            else:
                run_id, answer, metadata = await runner.run(corpus_request, history)
        except Exception as exc:
            observation = await self._failure_observation(
                request,
                corpus_request,
                history,
                identifier,
                fingerprint,
                effective,
                exc,
            )
            await self._persist(
                observation,
                {"error": type(exc).__name__},
                await self._is_evaluation_corpus_id(request.corpus_id),
            )
            status = "timeout" if isinstance(exc, TimeoutError) else "failed"
            raise EvalServiceError(
                "evaluation_timeout" if status == "timeout" else "evaluation_failed",
                "The evaluation request did not complete successfully.",
                504 if status == "timeout" else 502,
            ) from exc
        chunks = await self._hydrate_chunks(metadata)
        observation = self._builder.from_basic_rag_metadata(
            run_id=run_id,
            request_id=identifier,
            corpus_id=request.corpus_id,
            architecture_name=self._architecture_name,
            architecture_version=self._architecture_version,
            request=corpus_request,
            chunks=chunks,
            answer=answer,
            metadata=metadata,
            effective_configuration=effective,
            history=history,
        )
        evaluation_corpus = await self._evaluation_corpus(request.corpus_id)
        observation = observation.model_copy(
            update={
                "request": observation.request.model_copy(
                    update={
                        "conversation_history": request.history,
                        "included_history_turn_ids": [
                            str(item["id"])
                            for item in request.history
                            if isinstance(item, dict) and item.get("id") is not None
                        ],
                        "filters": request.filters,
                        "parameters": {
                            **request.parameters,
                            "config_overrides": overrides,
                        },
                    }
                ),
                "completed_at": datetime.now(UTC),
                "conversation_id": request.conversation_id,
                "turn_id": identifier,
                "turn_index": request.turn_index,
                "conversation_state": {
                    "prior_turn_ids": [
                        str(item["id"])
                        for item in request.history
                        if isinstance(item, dict) and item.get("id") is not None
                    ],
                    "included_history_turn_ids": [
                        str(item["id"])
                        for item in request.history
                        if isinstance(item, dict) and item.get("id") is not None
                    ],
                },
                "evaluation_corpus": evaluation_corpus,
                "benchmark_case_id": request.benchmark_case_id,
                "experiment_id": request.experiment_id,
                "seed": request.seed,
                "perturbation_metadata": request.perturbation_metadata,
                "request_fingerprint": fingerprint,
            }
        )
        await self._persist(observation, metadata, evaluation_corpus is not None)
        return observation

    async def retrieve(
        self,
        request: EvalRetrieveRequest,
        *,
        request_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> RagRunObservation:
        """Run candidate/final-context retrieval without invoking generation."""
        fingerprint = self._fingerprint("retrieve", request)
        identifier = (
            request_id or idempotency_key or self._deterministic_id(fingerprint)
        )
        existing = await self._existing(identifier, fingerprint)
        if existing is not None:
            return existing
        runner = self._runner()
        overrides = self._overrides(request)
        effective = self._effective(request.corpus_id, overrides)
        corpus_request = CorpusRequest(
            request.corpus_id,
            request.query,
            document_ids=self._document_ids(request.filters),
        )
        try:
            if overrides:
                chunks = await runner.retrieve(corpus_request, runtime_config=overrides)
            else:
                chunks = await runner.retrieve(corpus_request)
        except Exception as exc:
            history = self._history(request.history)
            observation = await self._failure_observation(
                request,
                corpus_request,
                history,
                identifier,
                fingerprint,
                effective,
                exc,
            )
            await self._persist(
                observation,
                {"error": type(exc).__name__},
                await self._is_evaluation_corpus_id(request.corpus_id),
            )
            status = "timeout" if isinstance(exc, TimeoutError) else "failed"
            raise EvalServiceError(
                "evaluation_timeout" if status == "timeout" else "evaluation_failed",
                "The evaluation request did not complete successfully.",
                504 if status == "timeout" else 502,
            ) from exc
        evaluation_corpus = await self._evaluation_corpus(request.corpus_id)
        observation = self._builder.build(
            run_id=str(uuid4()),
            request_id=identifier,
            corpus_id=request.corpus_id,
            architecture_name=self._architecture_name,
            architecture_version=self._architecture_version,
            request=corpus_request,
            chunks=chunks,
            effective_configuration=effective,
        ).model_copy(
            update={
                "request": {
                    "query": request.query,
                    "history": request.history,
                    "filters": request.filters,
                    "parameters": {
                        **request.parameters,
                        "config_overrides": overrides,
                    },
                },
                "answer": None,
                "status": "retrieved",
                "completed_at": datetime.now(UTC),
                "conversation_id": request.conversation_id,
                "turn_id": identifier,
                "turn_index": request.turn_index,
                "evaluation_corpus": evaluation_corpus,
                "benchmark_case_id": request.benchmark_case_id,
                "experiment_id": request.experiment_id,
                "seed": request.seed,
                "perturbation_metadata": request.perturbation_metadata,
                "request_fingerprint": fingerprint,
            }
        )
        await self._persist(observation, {}, evaluation_corpus is not None)
        return observation

    async def by_request_id(self, request_id: str) -> RagRunObservation:
        """Restore a persisted observation or raise a protocol 404 error."""
        existing = await self._existing(request_id)
        if existing is None:
            raise EvalServiceError(
                "request_not_found", "No observation has that request_id.", 404
            )
        return existing

    async def create_corpus(self, **kwargs: object):
        """Create an isolated evaluation corpus inventory."""
        return await self._corpora.create(**kwargs)

    async def get_corpus(self, corpus_id: str):
        """Load an isolated evaluation corpus inventory."""
        try:
            return await self._corpora.get(corpus_id)
        except KeyError as exc:
            raise EvalServiceError("corpus_not_found", str(exc), 404) from exc

    async def delete_corpus(self, corpus_id: str) -> None:
        """Delete an isolated evaluation corpus."""
        try:
            await self._corpora.delete(corpus_id)
        except KeyError as exc:
            raise EvalServiceError("corpus_not_found", str(exc), 404) from exc

    async def upload_corpus_documents(
        self, corpus_id: str, documents: list[dict[str, object]]
    ) -> dict[str, object]:
        """Append isolated source-document inventory."""
        try:
            return await self._corpora.upload_documents(corpus_id, documents)
        except KeyError as exc:
            raise EvalServiceError("corpus_not_found", str(exc), 404) from exc

    async def upload_corpus_chunks(
        self, corpus_id: str, chunks: list[dict[str, object]]
    ) -> dict[str, object]:
        """Append isolated canonical chunk inventory."""
        try:
            return await self._corpora.upload_chunks(corpus_id, chunks)
        except KeyError as exc:
            raise EvalServiceError("corpus_not_found", str(exc), 404) from exc

    async def operation(self, operation_id: str) -> dict[str, object]:
        """Load an isolated corpus operation status."""
        try:
            return await self._corpora.operation(operation_id)
        except KeyError as exc:
            raise EvalServiceError("operation_not_found", str(exc), 404) from exc

    async def _failure_observation(
        self,
        request: EvalQueryRequest,
        corpus_request: CorpusRequest,
        history: tuple[Message, ...],
        request_id: str,
        fingerprint: str,
        effective: object,
        error: Exception,
    ) -> RagRunObservation:
        """Persist a failed or timed-out attempt with its request identity."""
        status = "timeout" if isinstance(error, TimeoutError) else "failed"
        observation = self._builder.build(
            run_id=str(uuid4()),
            request_id=request_id,
            corpus_id=request.corpus_id,
            architecture_name=self._architecture_name,
            architecture_version=self._architecture_version,
            request=corpus_request,
            history=history,
            effective_configuration=effective,
            status=status,
            error=type(error).__name__,
        )
        return observation.model_copy(
            update={
                "request_fingerprint": fingerprint,
                "completed_at": datetime.now(UTC),
                "benchmark_case_id": request.benchmark_case_id,
                "experiment_id": request.experiment_id,
                "seed": request.seed,
                "perturbation_metadata": request.perturbation_metadata,
            }
        )

    async def _is_evaluation_corpus_id(self, corpus_id: str) -> bool:
        """Identify isolated IDs without treating production projects as eval data."""
        try:
            await self._corpora.get(corpus_id)
        except KeyError:
            return False
        return True

    def _runner(self):
        """Return the configured runner or a clear composition error."""
        if self._application.runner is None:
            raise EvalServiceError(
                "service_unavailable",
                "The evaluation runner is not configured.",
                503,
            )
        return self._application.runner

    @property
    def _architecture_name(self) -> str:
        """Return configured architecture identity."""
        return (
            self._application.architecture.name
            if self._application.architecture
            else "query_basic_v1"
        )

    @property
    def _architecture_version(self) -> int:
        """Return configured architecture version."""
        return (
            self._application.architecture.version
            if self._application.architecture
            else 1
        )

    @staticmethod
    def _overrides(request: EvalQueryRequest) -> dict[str, object]:
        """Merge structured settings over legacy flat evaluation overrides."""
        return {**request.config_overrides, **request.config}

    def _effective(self, corpus_id: str, overrides: dict[str, object]):
        """Build effective configuration when the composition root supplied it."""
        if not self._application.architecture or not self._application.ingestion_config:
            if overrides:
                raise EvalServiceError(
                    "config_unavailable",
                    "Per-request overrides require typed application configuration.",
                    503,
                )
            return None
        try:
            if not overrides:
                return build_effective_configuration(
                    self._application.architecture,
                    self._application.ingestion_config,
                    corpus_id=corpus_id,
                )
            return self._config.validate(
                self._application.architecture,
                self._application.ingestion_config,
                overrides,
                corpus_id=corpus_id,
            ).effective
        except ValueError as exc:
            raise EvalServiceError("unsupported_config", str(exc), 422) from exc

    async def _existing(
        self, request_id: str, fingerprint: str | None = None
    ) -> RagRunObservation | None:
        """Load an idempotent result when the configured repository supports it."""
        loader = getattr(
            self._application.repository, "observation_by_request_id", None
        )
        if loader is None:
            return None
        existing = await loader(request_id)
        if existing is not None and fingerprint is not None:
            if existing.request_fingerprint not in (None, fingerprint):
                raise EvalServiceError(
                    "idempotency_conflict",
                    "The request identity was already used for different input.",
                    409,
                )
        return existing

    @staticmethod
    def _fingerprint(operation: str, request: EvalQueryRequest) -> str:
        """Hash the complete request so idempotency keys cannot alias inputs."""
        payload = {"operation": operation, **request.model_dump(mode="json")}
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return sha256(canonical.encode()).hexdigest()

    @staticmethod
    def _deterministic_id(fingerprint: str) -> str:
        """Use the canonical request fingerprint as the generated request ID."""
        return fingerprint

    async def _persist(
        self,
        observation: RagRunObservation,
        metadata: dict[str, object],
        isolated_corpus: bool = False,
    ) -> None:
        """Persist both the run index and canonical observation when available."""
        record_run = getattr(self._application.repository, "record_run", None)
        if record_run is not None:
            await record_run(
                observation.run_id,
                None if isolated_corpus else observation.corpus_id,
                observation.architecture_name,
                int(observation.architecture_version),
                observation.status,
                metadata,
            )
        saver = getattr(self._application.repository, "save_observation", None)
        if saver is not None:
            await saver(observation)

    async def _evaluation_corpus(self, corpus_id: str):
        """Return an isolated corpus reference when this ID belongs to one."""
        try:
            return await self._corpora.get(corpus_id)
        except KeyError:
            return None

    async def _hydrate_chunks(
        self, metadata: dict[str, object]
    ) -> tuple[ChunkRetrievalResult, ...]:
        """Hydrate canonical chunks while retaining query rank and score."""
        entries = metadata.get("retrieval_results", [])
        if not isinstance(entries, list):
            return ()
        specs = [entry for entry in entries if isinstance(entry, dict)]
        ids = tuple(str(entry["chunk_id"]) for entry in specs if "chunk_id" in entry)
        chunks = await self._application.repository.chunks_by_ids(ids)
        return tuple(
            ChunkRetrievalResult(
                chunks[chunk_id],
                int(entry.get("rank", index)),
                str(entry.get("retriever_id", "unknown")),
                (
                    float(entry["score"])
                    if isinstance(entry.get("score"), (int, float))
                    else None
                ),
            )
            for index, entry in enumerate(specs, 1)
            if (chunk_id := str(entry.get("chunk_id", ""))) in chunks
        )

    @staticmethod
    def _document_ids(filters: dict[str, object]) -> tuple[str, ...]:
        """Extract the optional document filter without accepting arbitrary fields."""
        value = filters.get("document_ids", [])
        return (
            tuple(item for item in value if isinstance(item, str))
            if isinstance(value, list)
            else ()
        )

    @staticmethod
    def _history(history: list[dict[str, object]]) -> tuple[Message, ...]:
        """Convert protocol history into the existing domain message type."""
        return tuple(
            Message(
                str(uuid4()),
                "eval",
                str(item.get("role", "user")),
                str(item.get("content", "")),
                datetime.now(UTC),
            )
            for item in history
        )
