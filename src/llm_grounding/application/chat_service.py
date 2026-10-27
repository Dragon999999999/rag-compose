"""Thin persistent chat use cases that consume a project corpus through Basic RAG v1."""

from dataclasses import replace

from llm_grounding.application.pipeline_compatibility import (
    CompatibilityResult,
    IndexCapabilities,
    validate_query_compatibility,
)
from llm_grounding.application.pipeline_runtime import PipelineRuntimeResolver
from llm_grounding.application.pipeline_service import PipelineService
from llm_grounding.domain.answers import Answer
from llm_grounding.domain.corpus import Chat, CorpusRequest, Message, Pipeline, TaskType
from llm_grounding.infrastructure.config.models import PipelineConfig
from llm_grounding.infrastructure.database.repositories import PostgresRepository
from llm_grounding.orchestration.basic_rag import BasicRagRunner
from llm_grounding.reasoning.basic_grounded import GroundedResponsePayload

DEFAULT_CHAT_NAME = "New chat"


class PipelineSelectionError(ValueError):
    """Report a safe, client-facing failure to select a conversation pipeline."""

    def __init__(self, code: str, message: str, status_code: int = 409) -> None:
        """Retain a stable error code and HTTP status for API adapters."""
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def _fallback_turn_title(message: str) -> str:
    """Create a stable label when no structured model response is available."""
    normalized = " ".join(message.split())
    if len(normalized) <= 80:
        return normalized or "Untitled conversation turn"
    return f"{normalized[:77].rstrip()}…"


def _turn_title(answer: Answer, message: str) -> str:
    """Read the model-generated turn title or label a non-model fallback answer."""
    if answer.payload:
        payload = GroundedResponsePayload.model_validate(answer.payload)
        if "turn_title" in answer.payload:
            return payload.turn_title
    return _fallback_turn_title(message)


def _generated_chat_name(answer: Answer, run_metadata: dict[str, object]) -> str | None:
    """Read a validated first-turn name from accepted or rejected generation."""
    if not answer.payload:
        value = run_metadata.get("generated_chat_name")
        return value if isinstance(value, str) else None
    return GroundedResponsePayload.model_validate(answer.payload).chat_name


class ChatService:
    """Persist project-scoped messages without treating history as evidence."""

    def __init__(
        self,
        repository: PostgresRepository,
        runner: BasicRagRunner | None,
        max_history: int = 10,
        pipeline_runtime: PipelineRuntimeResolver | None = None,
        pipeline_service: PipelineService | None = None,
    ) -> None:
        """Inject persistence, query architecture, and bounded history policy."""
        self._repository, self._runner, self._max_history, self._pipeline_runtime = (
            repository,
            runner,
            max_history,
            pipeline_runtime,
        )
        self._pipeline_service = pipeline_service

    async def create_chat(
        self, project_id: str, name: str, pipeline_id: str | None = None
    ) -> Chat:
        """Create a conversation with an explicit or application-default pipeline."""
        normalized_name = name.strip()
        if self._pipeline_service is None:
            if pipeline_id is not None:
                raise PipelineSelectionError(
                    "pipeline_unavailable",
                    "Pipeline selection is unavailable in this application.",
                    503,
                )
            return await self._repository.create_chat(
                project_id,
                normalized_name,
                chat_name_set=(
                    normalized_name.casefold() != DEFAULT_CHAT_NAME.casefold()
                ),
            )
        selected = await self._select_pipeline(pipeline_id)
        indexed_representation = await self._validate_project_pipeline(
            project_id, selected
        )
        return await self._repository.create_chat(
            project_id,
            normalized_name,
            chat_name_set=normalized_name.casefold() != DEFAULT_CHAT_NAME.casefold(),
            pipeline_id=selected.id,
            pipeline_version=selected.version,
            pipeline_config_hash=selected.config_hash,
            pipeline_representation_hash=indexed_representation,
        )

    async def list_chats(self, project_id: str) -> tuple[Chat, ...]:
        """List chats belonging to a reusable project corpus."""
        return await self._repository.list_chats(project_id)

    async def delete_chat(self, chat_id: str) -> None:
        """Delete one conversation while retaining its project's corpus."""
        await self._repository.delete_chat(chat_id)

    async def rename_chat(self, chat_id: str, name: str) -> Chat:
        """Rename one project-scoped conversation."""
        return await self._repository.rename_chat(chat_id, name)

    async def set_pipeline(self, chat_id: str, pipeline_id: str) -> Chat:
        """Switch a conversation to a compatible active pipeline atomically.

        Query-time changes are allowed because the selected representation hash
        remains stable. Representation changes are rejected before persistence so
        an existing conversation cannot silently query an incompatible index.
        """
        if self._pipeline_service is None:
            raise PipelineSelectionError(
                "pipeline_unavailable",
                "Pipeline selection is unavailable in this application.",
                503,
            )
        chat = await self._repository.get_chat(chat_id)
        target = await self._select_pipeline(pipeline_id)
        indexed_representation = await self._validate_project_pipeline(
            chat.project_id, target
        )
        if target.kind != "query":
            raise PipelineSelectionError(
                "unsupported_pipeline",
                "Chat conversations require a query pipeline.",
                422,
            )
        try:
            return await self._repository.set_chat_pipeline(
                chat_id,
                target.id,
                pipeline_version=target.version,
                pipeline_config_hash=target.config_hash,
                pipeline_representation_hash=indexed_representation,
            )
        except KeyError as exc:
            raise PipelineSelectionError("chat_not_found", str(exc), 404) from exc
        except ValueError as exc:
            raise PipelineSelectionError("pipeline_changed", str(exc), 409) from exc

    async def messages(self, chat_id: str) -> tuple[Message, ...]:
        """Load the complete persisted conversation in chronological order."""
        return await self._repository.recent_messages(chat_id, 10_000)

    async def ask(
        self,
        chat_id: str,
        message: str,
        task_type: TaskType = TaskType.QA,
        pipeline_id: str | None = None,
    ) -> Answer:
        """Persist a question, query its corpus, then persist the answer."""
        chat = await self._repository.get_chat(chat_id)
        if pipeline_id is not None:
            chat = await self.set_pipeline(chat_id, pipeline_id)
        selected_runner = self._runner
        selected_name = "query_basic_v1"
        selected_version = 1
        selected_pipeline: Pipeline | None = None
        if self._pipeline_service is not None:
            selected_pipeline = await self._pipeline_for_chat(chat)
            if self._pipeline_runtime is None:
                raise PipelineSelectionError(
                    "runtime_unavailable",
                    "Pipeline runtime resolution is unavailable.",
                    503,
                )
            ingestion = await self._project_ingestion(chat.project_id)
            runtime = await self._pipeline_runtime.resolve(
                selected_pipeline.id, ingestion=ingestion
            )
            if not isinstance(runtime.runner, BasicRagRunner):
                raise PipelineSelectionError(
                    "unsupported_runtime",
                    "The selected pipeline is not a project-scoped Basic RAG runtime.",
                    422,
                )
            selected_runner = runtime.runner
            selected_name = runtime.architecture.name
            selected_version = runtime.architecture.version
        if selected_runner is None:
            raise PipelineSelectionError(
                "runtime_unavailable",
                "This project has no dense vector runtime; select a compatible "
                "query pipeline after building a dense index.",
                409,
            )
        generate_chat_name = not chat.chat_name_set
        await self._repository.add_message(chat_id, "user", message)
        history = await self._repository.recent_messages(chat_id, self._max_history)
        run_id, answer, metadata = await selected_runner.run(
            CorpusRequest(chat.project_id, message, task_type),
            history,
            generate_chat_name=generate_chat_name,
        )
        if selected_pipeline is None:
            await self._repository.record_run(
                run_id,
                chat.project_id,
                selected_name,
                selected_version,
                str(metadata["status"]),
                metadata,
            )
        else:
            pipeline_metadata = {
                **metadata,
                "pipeline_id": selected_pipeline.id,
                "pipeline_version": selected_pipeline.version,
                "pipeline_config_hash": selected_pipeline.config_hash,
                "pipeline_representation_hash": (
                    chat.pipeline_representation_hash
                    or selected_pipeline.representation_hash
                ),
            }
            await self._repository.record_run(
                run_id,
                chat.project_id,
                selected_name,
                selected_version,
                str(metadata["status"]),
                pipeline_metadata,
                pipeline_id=selected_pipeline.id,
                pipeline_version=selected_pipeline.version,
                pipeline_config_hash=selected_pipeline.config_hash,
                pipeline_representation_hash=(
                    chat.pipeline_representation_hash
                    or selected_pipeline.representation_hash
                ),
            )
        turn_title = _turn_title(answer, message)
        assistant_message = await self._repository.add_message(
            chat_id,
            "assistant",
            answer.text,
            run_id,
            turn_title,
        )
        if generate_chat_name and (chat_name := _generated_chat_name(answer, metadata)):
            await self._repository.set_generated_chat_name(chat_id, chat_name)
        retrieved = metadata.get("retrieved_chunk_ids", [])
        retrieved_ids = retrieved if isinstance(retrieved, list) else []
        return replace(
            answer,
            metadata={
                **answer.metadata,
                "rag_run_id": run_id,
                "message_id": assistant_message.id,
                "turn_title": assistant_message.turn_title,
                "retrieved_chunk_ids": [
                    str(chunk_id)
                    for chunk_id in retrieved_ids
                    if isinstance(chunk_id, str)
                ],
                "conversation_id": chat.id,
                "turn_id": assistant_message.id,
                "turn_index": len(
                    [item for item in history if item.role == "assistant"]
                ),
                "included_history_turn_ids": [item.id for item in history],
            },
        )

    async def _select_pipeline(self, pipeline_id: str | None) -> Pipeline:
        """Resolve and normalize an explicit or default pipeline selection."""
        assert self._pipeline_service is not None
        try:
            return await self._pipeline_service.select(pipeline_id, kind="query")
        except KeyError as exc:
            code = (
                "pipeline_not_found"
                if pipeline_id is not None
                else "default_unavailable"
            )
            status = 404 if pipeline_id is not None else 503
            raise PipelineSelectionError(code, str(exc), status) from exc
        except ValueError as exc:
            if "not query" in str(exc):
                raise PipelineSelectionError(
                    "unsupported_pipeline",
                    "Chat conversations require a query pipeline.",
                    422,
                ) from exc
            raise PipelineSelectionError("pipeline_inactive", str(exc), 409) from exc

    async def _pipeline_for_chat(self, chat: Chat) -> Pipeline:
        """Resolve a chat selection and reject silent changes to its snapshot."""
        assert self._pipeline_service is not None
        if chat.pipeline_id is None:
            selected = await self._select_pipeline(None)
            indexed_representation = await self._validate_project_pipeline(
                chat.project_id, selected
            )
            try:
                await self._repository.set_chat_pipeline(
                    chat.id,
                    selected.id,
                    pipeline_version=selected.version,
                    pipeline_config_hash=selected.config_hash,
                    pipeline_representation_hash=indexed_representation,
                )
            except (KeyError, ValueError) as exc:
                raise PipelineSelectionError("pipeline_changed", str(exc), 409) from exc
            return selected
        try:
            selected = await self._pipeline_service.select(
                chat.pipeline_id, kind="query"
            )
        except KeyError as exc:
            raise PipelineSelectionError("pipeline_not_found", str(exc), 404) from exc
        except ValueError as exc:
            code = "pipeline_inactive" if "inactive" in str(exc) else "pipeline_changed"
            raise PipelineSelectionError(code, str(exc), 409) from exc
        if not selected.active:
            raise PipelineSelectionError(
                "pipeline_inactive",
                f"The conversation's pipeline {selected.name!r} is inactive.",
                409,
            )
        if chat.pipeline_config_hash is not None and (
            selected.config_hash != chat.pipeline_config_hash
            or selected.version != chat.pipeline_version
            or selected.kind != "query"
        ):
            raise PipelineSelectionError(
                "pipeline_configuration_changed",
                "The conversation's pipeline configuration changed; explicitly "
                "select a pipeline again before continuing.",
                409,
            )
        await self._validate_project_pipeline(chat.project_id, selected)
        return selected

    async def _project_ingestion(self, project_id: str):
        """Resolve the effective ingestion configuration selected by a project."""
        if self._pipeline_service is None:
            return None
        get_project = getattr(self._repository, "get_project", None)
        if get_project is None:
            return None
        project = await get_project(project_id)
        if project.ingestion_pipeline_id is None:
            raise PipelineSelectionError(
                "ingestion_unavailable",
                "The project has no selected ingestion pipeline.",
                409,
            )
        return await self._pipeline_service.resolve_ingestion(
            project.ingestion_pipeline_id
        )

    async def _validate_project_pipeline(
        self, project_id: str, pipeline: Pipeline
    ) -> str:
        """Validate a query pipeline against the project's indexed capabilities."""
        if self._pipeline_service is None:
            return pipeline.representation_hash
        get_project = getattr(self._repository, "get_project", None)
        if get_project is None:
            # Legacy repository adapters do not expose project index metadata.
            return pipeline.representation_hash
        project = await get_project(project_id)
        config = PipelineConfig.model_validate(pipeline.config)
        index = IndexCapabilities.from_mapping(project.index_capabilities)
        compatibility: CompatibilityResult = validate_query_compatibility(
            config,
            index,
            self._pipeline_service.registry,
            self._pipeline_service.model_catalog,
        )
        if pipeline.kind != "query":
            raise PipelineSelectionError(
                "unsupported_pipeline",
                "Chat conversations require a query pipeline.",
                422,
            )
        if not compatibility.compatible:
            missing = ", ".join(sorted(compatibility.missing_capabilities))
            suffix = f" Missing: {missing}." if missing else ""
            raise PipelineSelectionError(
                "incompatible_index",
                (
                    compatibility.reason
                    or "The pipeline is incompatible with the project index"
                )
                + suffix,
            )
        return project.ingestion_representation_hash or pipeline.representation_hash
