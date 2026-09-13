"""Thin persistent chat use cases that consume a project corpus through Basic RAG v1."""

from dataclasses import replace

from llm_grounding.domain.answers import Answer
from llm_grounding.domain.corpus import Chat, CorpusRequest, Message, TaskType
from llm_grounding.infrastructure.database.repositories import PostgresRepository
from llm_grounding.orchestration.basic_rag import BasicRagRunner
from llm_grounding.reasoning.basic_grounded import GroundedResponsePayload

DEFAULT_CHAT_NAME = "New chat"


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
        runner: BasicRagRunner,
        max_history: int = 10,
    ) -> None:
        """Inject persistence, query architecture, and bounded history policy."""
        self._repository, self._runner, self._max_history = (
            repository,
            runner,
            max_history,
        )

    async def create_chat(self, project_id: str, name: str) -> Chat:
        """Create a project-scoped conversation."""
        normalized_name = name.strip()
        return await self._repository.create_chat(
            project_id,
            normalized_name,
            chat_name_set=normalized_name.casefold() != DEFAULT_CHAT_NAME.casefold(),
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

    async def messages(self, chat_id: str) -> tuple[Message, ...]:
        """Load the complete persisted conversation in chronological order."""
        return await self._repository.recent_messages(chat_id, 10_000)

    async def ask(
        self, chat_id: str, message: str, task_type: TaskType = TaskType.QA
    ) -> Answer:
        """Persist a question, query its corpus, then persist the answer."""
        chat = await self._repository.get_chat(chat_id)
        generate_chat_name = not chat.chat_name_set
        await self._repository.add_message(chat_id, "user", message)
        history = await self._repository.recent_messages(chat_id, self._max_history)
        run_id, answer, metadata = await self._runner.run(
            CorpusRequest(chat.project_id, message, task_type),
            history,
            generate_chat_name=generate_chat_name,
        )
        await self._repository.record_run(
            run_id,
            chat.project_id,
            "basic_rag_v1",
            1,
            str(metadata["status"]),
            metadata,
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
            },
        )
