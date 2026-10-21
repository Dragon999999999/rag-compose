"""Test persistent conversation pipeline selection and safe switching."""

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from llm_grounding.api.main import create_api
from llm_grounding.application.chat_service import (
    ChatService,
    PipelineSelectionError,
)
from llm_grounding.application.pipeline_service import PipelineService
from llm_grounding.domain.answers import Answer
from llm_grounding.domain.corpus import Chat, Message, Pipeline
from llm_grounding.infrastructure.config.loader import (
    load_model_catalog,
    load_pipeline_config,
)
from llm_grounding.infrastructure.config.models import ModelCatalog, PipelineConfig
from llm_grounding.orchestration.basic_rag import BasicRagRunner


def pipeline_records() -> tuple[Pipeline, ...]:
    """Build one default, one compatible, and one incompatible pipeline."""
    catalog = load_model_catalog(Path("configs/models.yaml"))
    base = load_pipeline_config(
        Path("configs/pipelines/basic_rag_v1.yaml"), catalog=catalog
    )
    compatible = base.model_copy(
        update={
            "name": "compatible_pipeline",
            "is_default": False,
            "query": base.query.model_copy(
                update={
                    "retriever": base.query.retriever.model_copy(update={"top_k": 4})
                }
            ),
        }
    )
    incompatible = base.model_copy(
        update={
            "name": "incompatible_pipeline",
            "is_default": False,
            "representation": base.representation.model_copy(
                update={
                    "chunker": base.representation.chunker.model_copy(
                        update={"target_chars": 600}
                    )
                }
            ),
        }
    )
    return tuple(
        _record(identifier, config, catalog)
        for identifier, config in (
            ("pipeline-a", base),
            ("pipeline-b", compatible),
            ("pipeline-c", incompatible),
        )
    )


def _record(identifier: str, config: PipelineConfig, catalog: ModelCatalog) -> Pipeline:
    """Create a detached pipeline row for service-level tests."""
    now = datetime.now(UTC)
    return Pipeline(
        identifier,
        config.name,
        config.version,
        config.schema_version,
        config.description,
        config.model_dump(mode="json"),
        config.config_hash,
        config.representation_hash(catalog),
        "api",
        None,
        None,
        True,
        config.is_default,
        now,
        now,
    )


class ChatPipelineRepository:
    """Keep chats, messages, and pipeline records in deterministic memory."""

    def __init__(self, records: tuple[Pipeline, ...]) -> None:
        """Initialize pipeline records and an empty project conversation."""
        self.pipeline_records = {record.id: record for record in records}
        self.chat = Chat("chat", "project", "New chat", datetime.now(UTC))
        self.messages: list[Message] = []
        self.run_records: list[dict[str, object]] = []

    async def list_pipelines(
        self, *, active_only: bool = False
    ) -> tuple[Pipeline, ...]:
        """List stored pipelines, optionally excluding deactivated records."""
        values = tuple(self.pipeline_records.values())
        return tuple(record for record in values if record.active or not active_only)

    async def get_pipeline(self, pipeline_id: str) -> Pipeline:
        """Return one pipeline record."""
        return self.pipeline_records[pipeline_id]

    async def get_pipeline_by_name(self, name: str) -> Pipeline | None:
        """Find a pipeline by stable name."""
        return next(
            (
                record
                for record in self.pipeline_records.values()
                if record.name == name
            ),
            None,
        )

    async def get_chat(self, chat_id: str) -> Chat:
        """Return the only test conversation."""
        assert chat_id == self.chat.id
        return self.chat

    async def create_chat(
        self,
        project_id: str,
        name: str,
        chat_name_set: bool = False,
        *,
        pipeline_id: str | None = None,
        pipeline_version: int | None = None,
        pipeline_config_hash: str | None = None,
        pipeline_representation_hash: str | None = None,
    ) -> Chat:
        """Persist a chat and its immutable pipeline selection snapshot."""
        self.chat = Chat(
            "chat",
            project_id,
            name,
            self.chat.created_at,
            chat_name_set,
            pipeline_id=pipeline_id,
            pipeline_version=pipeline_version,
            pipeline_config_hash=pipeline_config_hash,
            pipeline_representation_hash=pipeline_representation_hash,
        )
        return self.chat

    async def set_chat_pipeline(
        self,
        chat_id: str,
        pipeline_id: str,
        *,
        pipeline_version: int,
        pipeline_config_hash: str,
        pipeline_representation_hash: str,
    ) -> Chat:
        """Atomically replace the chat selection after checking its target snapshot."""
        assert chat_id == self.chat.id
        target = self.pipeline_records[pipeline_id]
        if not target.active:
            raise ValueError("pipeline inactive")
        if (
            target.version != pipeline_version
            or target.config_hash != pipeline_config_hash
            or target.representation_hash != pipeline_representation_hash
        ):
            raise ValueError("pipeline changed")
        self.chat = replace(
            self.chat,
            pipeline_id=target.id,
            pipeline_version=target.version,
            pipeline_config_hash=target.config_hash,
            pipeline_representation_hash=target.representation_hash,
        )
        return self.chat

    async def rename_chat(self, chat_id: str, name: str) -> Chat:
        """Rename the test conversation without changing pipeline selection."""
        self.chat = replace(self.chat, name=name, chat_name_set=True)
        return self.chat

    async def list_chats(self, project_id: str) -> tuple[Chat, ...]:
        """List the persisted test conversation for the project."""
        assert project_id == self.chat.project_id
        return (self.chat,)

    async def add_message(
        self,
        chat_id: str,
        role: str,
        content: str,
        rag_run_id: str | None = None,
        turn_title: str | None = None,
    ) -> Message:
        """Persist one test message."""
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
        """Return recent test messages in chronological order."""
        return tuple(self.messages[-limit:])

    async def record_run(self, *args: object, **kwargs: object) -> None:
        """Record run arguments so pipeline provenance can be asserted."""
        self.run_records.append({"args": args, "kwargs": kwargs})

    async def set_generated_chat_name(self, chat_id: str, name: str) -> Chat:
        """Apply a generated name without changing the pipeline."""
        self.chat = replace(self.chat, name=name, chat_name_set=True)
        return self.chat


class RecordingRuntime:
    """Return a Basic RAG-shaped runtime and record selected pipeline IDs."""

    def __init__(self, repository: ChatPipelineRepository) -> None:
        """Create one test runner accepted by ChatService."""
        self.repository = repository
        self.resolved: list[str] = []
        self.invalidated: list[str] = []
        self.runner = object.__new__(BasicRagRunner)

        async def run(*args: object, **kwargs: object):
            """Return a deterministic answer for the selected runtime."""
            return (
                "run",
                Answer("answer"),
                {"status": "completed", "retrieved_chunk_ids": []},
            )

        self.runner.run = run  # type: ignore[method-assign]

    async def resolve(self, pipeline_id: str) -> SimpleNamespace:
        """Resolve a pipeline into the recording Basic RAG runner."""
        self.resolved.append(pipeline_id)
        record = await self.repository.get_pipeline(pipeline_id)
        return SimpleNamespace(
            record=record,
            runner=self.runner,
            architecture=SimpleNamespace(name=record.name, version=record.version),
        )

    def invalidate(self, pipeline_id: str) -> None:
        """Record cache invalidation caused by a pipeline update."""
        self.invalidated.append(pipeline_id)


async def test_chat_creation_persists_pipeline_for_messages() -> None:
    """Creation stores selection and subsequent messages resolve it automatically."""
    records = pipeline_records()
    repository = ChatPipelineRepository(records)
    service = PipelineService(repository, load_model_catalog())  # type: ignore[arg-type]
    runtime = RecordingRuntime(repository)
    chats = ChatService(
        repository,  # type: ignore[arg-type]
        runtime.runner,
        pipeline_runtime=runtime,  # type: ignore[arg-type]
        pipeline_service=service,  # type: ignore[arg-type]
    )

    created = await chats.create_chat("project", "Research", "pipeline-a")
    await chats.ask("chat", "What is the contribution?")

    assert created.pipeline_id == "pipeline-a"
    assert repository.chat.pipeline_id == "pipeline-a"
    assert runtime.resolved == ["pipeline-a"]
    assert repository.run_records[0]["kwargs"]["pipeline_id"] == "pipeline-a"
    assert (
        repository.run_records[0]["kwargs"]["pipeline_config_hash"]
        == records[0].config_hash
    )


async def test_compatible_switch_persists_and_incompatible_switch_is_atomic() -> None:
    """Query-only switching succeeds while representation changes stay atomic."""
    records = pipeline_records()
    repository = ChatPipelineRepository(records)
    service = PipelineService(repository, load_model_catalog())  # type: ignore[arg-type]
    runtime = RecordingRuntime(repository)
    chats = ChatService(
        repository,  # type: ignore[arg-type]
        runtime.runner,
        pipeline_runtime=runtime,  # type: ignore[arg-type]
        pipeline_service=service,  # type: ignore[arg-type]
    )
    await chats.create_chat("project", "Research", "pipeline-a")

    switched = await chats.set_pipeline("chat", "pipeline-b")
    assert switched.pipeline_id == "pipeline-b"
    before_rejected_switch = repository.chat

    with pytest.raises(PipelineSelectionError, match="incompatible") as caught:
        await chats.set_pipeline("chat", "pipeline-c")

    assert caught.value.code == "incompatible_representation"
    assert repository.chat == before_rejected_switch


async def test_pipeline_updates_and_deactivation_are_safe() -> None:
    """Updates never redirect a chat and inactive selections do not fall back."""
    records = pipeline_records()
    repository = ChatPipelineRepository(records)
    service = PipelineService(repository, load_model_catalog())  # type: ignore[arg-type]
    runtime = RecordingRuntime(repository)
    chats = ChatService(
        repository,  # type: ignore[arg-type]
        runtime.runner,
        pipeline_runtime=runtime,  # type: ignore[arg-type]
        pipeline_service=service,  # type: ignore[arg-type]
    )
    await chats.create_chat("project", "Research", "pipeline-b")
    current = repository.pipeline_records["pipeline-b"]
    changed_config = load_pipeline_config(
        Path("configs/pipelines/basic_rag_v1.yaml")
    ).model_copy(update={"name": current.name, "description": "updated"})
    repository.pipeline_records[current.id] = replace(
        current,
        config=changed_config.model_dump(mode="json"),
        config_hash=changed_config.config_hash,
    )

    with pytest.raises(PipelineSelectionError, match="changed"):
        await chats.ask("chat", "Continue")
    assert repository.chat.pipeline_config_hash == current.config_hash

    await chats.set_pipeline("chat", "pipeline-b")
    repository.pipeline_records["pipeline-b"] = replace(
        repository.pipeline_records["pipeline-b"], active=False
    )
    with pytest.raises(PipelineSelectionError, match="inactive"):
        await chats.ask("chat", "Continue again")
    assert repository.chat.pipeline_id == "pipeline-b"


async def test_chat_api_persists_selection_and_rejects_switch() -> None:
    """Chat endpoints expose selection and preserve atomic switch failures."""
    records = pipeline_records()
    repository = ChatPipelineRepository(records)
    service = PipelineService(repository, load_model_catalog())  # type: ignore[arg-type]
    runtime = RecordingRuntime(repository)
    chats = ChatService(
        repository,  # type: ignore[arg-type]
        runtime.runner,
        pipeline_runtime=runtime,  # type: ignore[arg-type]
        pipeline_service=service,  # type: ignore[arg-type]
    )
    application = SimpleNamespace(
        chats=chats,
        pipeline_service=service,
        evaluation=SimpleNamespace(corpus_service=object(), embedding_store=object()),
        repository=repository,
        _vectors=object(),
    )
    api = create_api(application)  # type: ignore[arg-type]

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api), base_url="http://test"
    ) as client:
        created = await client.post(
            "/api/projects/project/chats",
            json={"name": "Research", "pipeline_id": "pipeline-a"},
        )
        rejected = await client.patch(
            "/api/chats/chat", json={"pipeline_id": "pipeline-c"}
        )
        listed = await client.get("/api/projects/project/chats")

    assert created.status_code == 200
    assert created.json()["pipeline_id"] == "pipeline-a"
    assert rejected.status_code == 409
    assert rejected.json()["detail"]["code"] == "incompatible_representation"
    assert listed.json()[0]["pipeline_id"] == "pipeline-a"
