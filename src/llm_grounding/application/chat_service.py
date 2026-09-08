"""Thin persistent chat use cases that consume a project corpus through Basic RAG v1."""

from llm_grounding.domain.answers import Answer
from llm_grounding.domain.corpus import Chat, CorpusRequest, TaskType
from llm_grounding.infrastructure.database.repositories import PostgresRepository
from llm_grounding.orchestration.basic_rag import BasicRagRunner


class ChatService:
    """Persist messages around a project-scoped query runner without treating history as evidence."""

    def __init__(
        self,
        repository: PostgresRepository,
        runner: BasicRagRunner,
        max_history: int = 10,
    ) -> None:
        """Inject persistence, selected query architecture, and a bounded history policy."""
        self._repository, self._runner, self._max_history = (
            repository,
            runner,
            max_history,
        )

    async def create_chat(self, project_id: str, name: str) -> Chat:
        """Create a project-scoped conversation."""
        return await self._repository.create_chat(project_id, name)

    async def ask(
        self, chat_id: str, message: str, task_type: TaskType = TaskType.QA
    ) -> Answer:
        """Persist a question, query only its project's corpus, then persist the answer."""
        chat = await self._repository.get_chat(chat_id)
        await self._repository.add_message(chat_id, "user", message)
        history = await self._repository.recent_messages(chat_id, self._max_history)
        run_id, answer, metadata = await self._runner.run(
            CorpusRequest(chat.project_id, message, task_type), history
        )
        await self._repository.record_run(
            run_id,
            chat.project_id,
            "basic_rag_v1",
            1,
            str(metadata["status"]),
            metadata,
        )
        await self._repository.add_message(chat_id, "assistant", answer.text, run_id)
        return answer
