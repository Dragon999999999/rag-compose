"""Test chat naming behavior that follows grounded-answer generation."""

from datetime import UTC, datetime

from llm_grounding.application.chat_service import ChatService
from llm_grounding.domain.answers import Answer
from llm_grounding.domain.corpus import Chat, CorpusRequest, Message


class RecordingRepository:
    """Store the minimal state needed to exercise the chat application service."""

    def __init__(self) -> None:
        """Create a new default-named chat and empty message history."""
        self.chat = Chat("chat", "project", "New chat", datetime.now(UTC))
        self.messages: list[Message] = []
        self.renamed_to: str | None = None

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
        """Record the title assigned to the first grounded answer."""
        assert chat_id == self.chat.id
        self.renamed_to = name
        self.chat = Chat(self.chat.id, self.chat.project_id, name, self.chat.created_at)
        return self.chat


class FixedRunner:
    """Return a schema-compatible first answer without retrieval infrastructure."""

    async def run(
        self, request: CorpusRequest, history: tuple[Message, ...]
    ) -> tuple[str, Answer, dict[str, object]]:
        """Return a first-answer title representative of a grounded model response."""
        assert request.project_id == "project"
        assert len(history) == 1
        return (
            "run",
            Answer(
                "Insufficient evidence: answer.",
                payload={
                    "turn_title": "Paper contribution",
                    "answer_text": "answer.",
                    "claims": [],
                    "insufficient_evidence": True,
                    "limitations": None,
                },
            ),
            {"status": "completed", "retrieved_chunk_ids": []},
        )


async def test_first_answer_renames_default_chat_from_turn_title() -> None:
    """A default chat receives its generated topic title after the first answer."""
    repository = RecordingRepository()
    service = ChatService(repository, FixedRunner())  # type: ignore[arg-type]

    answer = await service.ask("chat", "What is the contribution?")

    assert repository.renamed_to == "Paper contribution"
    assert answer.metadata["turn_title"] == "Paper contribution"
