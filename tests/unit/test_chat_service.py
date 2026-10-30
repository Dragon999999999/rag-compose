"""Test chat naming behavior that follows grounded-answer generation."""

from datetime import UTC, datetime

from llm_grounding.application.chat_service import ChatService
from llm_grounding.domain.answers import Answer
from llm_grounding.domain.corpus import Chat, CorpusRequest, Message


class RecordingRepository:
    """Store the minimal state needed to exercise the chat application service."""

    def __init__(self) -> None:
        """Create a new default-named chat and empty message history."""
        self.chat = Chat("chat", "project", "New chat", datetime.now(UTC), False)
        self.messages: list[Message] = []
        self.generated_name_attempts: list[str] = []

    async def create_chat(
        self, project_id: str, name: str, chat_name_set: bool = False
    ) -> Chat:
        """Record initial name state selected by the application service."""
        self.chat = Chat("chat", project_id, name, datetime.now(UTC), chat_name_set)
        return self.chat

    async def get_chat(self, chat_id: str) -> Chat:
        """Return the configured chat."""
        assert chat_id == self.chat.id
        return self.chat

    async def add_message(
        self,
        chat_id: str,
        role: str,
        content: str,
        rag_run_id: str | None = None,
        turn_title: str | None = None,
    ) -> Message:
        """Record one message with stable test-time identifiers."""
        message = Message(
            f"message-{len(self.messages)}",
            chat_id,
            role,
            content,
            datetime.now(UTC),
            rag_run_id,
            turn_title,
        )
        self.messages.append(message)
        return message

    async def recent_messages(self, chat_id: str, limit: int) -> tuple[Message, ...]:
        """Return recorded history in insertion order."""
        assert chat_id == self.chat.id
        return tuple(self.messages[-limit:])

    async def record_run(self, *args: object) -> None:
        """Accept run persistence without introducing a database dependency."""

    async def rename_chat(self, chat_id: str, name: str) -> Chat:
        """Apply a manual rename and finalize the chat name."""
        assert chat_id == self.chat.id
        self.chat = Chat(
            self.chat.id,
            self.chat.project_id,
            name,
            self.chat.created_at,
            True,
        )
        return self.chat

    async def set_generated_chat_name(self, chat_id: str, name: str) -> Chat:
        """Apply a generated name only while the name remains unset."""
        assert chat_id == self.chat.id
        self.generated_name_attempts.append(name)
        if not self.chat.chat_name_set:
            self.chat = Chat(
                self.chat.id,
                self.chat.project_id,
                name,
                self.chat.created_at,
                True,
            )
        return self.chat


class FixedRunner:
    """Return a schema-compatible first answer without retrieval infrastructure."""

    def __init__(self) -> None:
        """Record whether each invocation requested an initial chat name."""
        self.generate_chat_name_values: list[bool] = []

    async def run(
        self,
        request: CorpusRequest,
        history: tuple[Message, ...],
        *,
        generate_chat_name: bool = False,
    ) -> tuple[str, Answer, dict[str, object]]:
        """Return a first-answer title representative of a grounded model response."""
        assert request.project_id == "project"
        assert len(history) == 1
        self.generate_chat_name_values.append(generate_chat_name)
        return (
            "run",
            Answer(
                "Insufficient evidence: answer.",
                payload={
                    "chat_name": "Grounded Paper Questions",
                    "turn_title": "Paper contribution",
                    "answer_text": "answer.",
                    "claims": [],
                    "insufficient_evidence": True,
                    "limitations": None,
                },
            ),
            {"status": "completed", "retrieved_chunk_ids": []},
        )


async def test_first_answer_sets_generated_chat_name_and_persisted_flag() -> None:
    """The first answer replaces a default name and finalizes its persisted state."""
    repository = RecordingRepository()
    runner = FixedRunner()
    service = ChatService(repository, runner)  # type: ignore[arg-type]

    answer = await service.ask("chat", "What is the contribution?")

    assert runner.generate_chat_name_values == [True]
    assert repository.generated_name_attempts == ["Grounded Paper Questions"]
    assert repository.chat.name == "Grounded Paper Questions"
    assert repository.chat.chat_name_set is True
    assert answer.metadata["turn_title"] == "Paper contribution"


async def test_manual_rename_to_default_text_prevents_generated_replacement() -> None:
    """A user-selected default name remains final despite matching the placeholder."""
    repository = RecordingRepository()
    runner = FixedRunner()
    service = ChatService(repository, runner)  # type: ignore[arg-type]

    renamed = await service.rename_chat("chat", "New Chat")
    await service.ask("chat", "What is the contribution?")

    assert renamed.chat_name_set is True
    assert runner.generate_chat_name_values == [False]
    assert repository.generated_name_attempts == []
    assert repository.chat.name == "New Chat"


class RenameDuringGenerationRunner(FixedRunner):
    """Simulate a manual rename completing while the first answer is generated."""

    def __init__(self, repository: RecordingRepository) -> None:
        """Retain the repository whose chat changes during generation."""
        super().__init__()
        self._repository = repository

    async def run(
        self,
        request: CorpusRequest,
        history: tuple[Message, ...],
        *,
        generate_chat_name: bool = False,
    ) -> tuple[str, Answer, dict[str, object]]:
        """Finalize a manual name before returning the generated suggestion."""
        self._repository.chat = Chat(
            self._repository.chat.id,
            self._repository.chat.project_id,
            "My chosen name",
            self._repository.chat.created_at,
            True,
        )
        return await super().run(
            request,
            history,
            generate_chat_name=generate_chat_name,
        )


async def test_manual_rename_during_generation_is_not_overwritten() -> None:
    """The post-generation state check preserves a concurrently chosen name."""
    repository = RecordingRepository()
    runner = RenameDuringGenerationRunner(repository)
    service = ChatService(repository, runner)  # type: ignore[arg-type]

    await service.ask("chat", "What is the contribution?")

    assert runner.generate_chat_name_values == [True]
    assert repository.generated_name_attempts == ["Grounded Paper Questions"]
    assert repository.chat.name == "My chosen name"
    assert repository.chat.chat_name_set is True


async def test_chat_creation_distinguishes_placeholder_from_explicit_name() -> None:
    """Only the automatic placeholder remains eligible for generated replacement."""
    repository = RecordingRepository()
    service = ChatService(repository, FixedRunner())  # type: ignore[arg-type]

    automatic = await service.create_chat("project", "New chat")
    explicit = await service.create_chat("project", "Literature review")

    assert automatic.chat_name_set is False
    assert explicit.chat_name_set is True
