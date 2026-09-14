"""Test persisted, grounded document summaries outside chat conversations."""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

from llm_grounding.application.document_summary_service import (
    DOCUMENT_SUMMARY_TASK,
    DocumentSummaryService,
)
from llm_grounding.domain.answers import Answer
from llm_grounding.domain.corpus import (
    Document,
    DocumentIngestion,
    IngestionStatus,
    SummaryStatus,
    TaskType,
)
from llm_grounding.reasoning.basic_grounded import GroundedResponsePayload


class SummaryRepository:
    """Retain one document summary using the repository boundary's contract."""

    def __init__(self) -> None:
        """Start with no generated overview."""
        self.summary = None
        self.recorded_runs: list[tuple[object, ...]] = []

    async def document_summary(self, document_id: str, ingestion_id: str):
        """Return the persisted document-summary state."""
        return self.summary

    async def page_count_for_ingestion(self, ingestion_id: str) -> int:
        """Provide a parsed PDF page count for the document overview."""
        return 24

    async def begin_document_summary(
        self, document_id: str, ingestion_id: str, page_count: int | None
    ):
        """Persist the generating lifecycle state."""
        from llm_grounding.domain.corpus import DocumentSummary

        now = datetime.now(UTC)
        self.summary = DocumentSummary(
            document_id,
            ingestion_id,
            SummaryStatus.GENERATING,
            None,
            None,
            page_count,
            now,
            now,
        )
        return self.summary

    async def record_run(self, *args: object) -> None:
        """Record the generated RAG run arguments."""
        self.recorded_runs.append(args)

    async def complete_document_summary(
        self, document_id: str, ingestion_id: str, rag_run_id: str
    ):
        """Persist a ready state for the successful grounded run."""
        assert self.summary is not None
        self.summary = self.summary.__class__(
            document_id,
            ingestion_id,
            SummaryStatus.READY,
            rag_run_id,
            None,
            self.summary.page_count,
            self.summary.created_at,
            datetime.now(UTC),
        )
        return self.summary

    async def fail_document_summary(
        self, document_id: str, ingestion_id: str, error: str
    ):
        """Persist a nonfatal summary-generation failure."""
        assert self.summary is not None
        self.summary = self.summary.__class__(
            document_id,
            ingestion_id,
            SummaryStatus.FAILED,
            None,
            error,
            self.summary.page_count,
            self.summary.created_at,
            datetime.now(UTC),
        )
        return self.summary


def _document() -> tuple[Document, DocumentIngestion]:
    """Create one ready document representation for summary tests."""
    now = datetime.now(UTC)
    return (
        Document("document", "paper.pdf", "digest", "pdf/digest.pdf", now),
        DocumentIngestion(
            "ingestion",
            "document",
            "config",
            "{}",
            "embedding",
            8,
            IngestionStatus.READY,
            now,
        ),
    )


async def test_document_summary_is_generated_once_and_persisted() -> None:
    """A ready summary is reused instead of issuing another model request."""
    repository = SummaryRepository()
    payload = GroundedResponsePayload(
        turn_title="Document overview",
        answer_text="The paper presents a supported contribution.",
        claims=[
            {"text": "The paper presents a contribution.", "source_chunk_ids": ["c1"]}
        ],
        insufficient_evidence=False,
    )
    runner = SimpleNamespace(
        run=AsyncMock(
            return_value=(
                "run",
                Answer("Rendered", payload=payload.model_dump()),
                {"status": "completed", "answer_payload": payload.model_dump()},
            )
        )
    )
    document, ingestion = _document()
    service = DocumentSummaryService(repository, runner)  # type: ignore[arg-type]

    first = await service.ensure_summary("project", document, ingestion)
    second = await service.ensure_summary("project", document, ingestion)

    assert first.status == SummaryStatus.READY
    assert second.rag_run_id == "run"
    runner.run.assert_awaited_once()
    request = runner.run.await_args.args[0]
    assert request.task == DOCUMENT_SUMMARY_TASK
    assert request.task_type == TaskType.SUMMARY
    assert request.document_ids == (document.id,)
    assert repository.recorded_runs[0][0] == "run"
    assert repository.recorded_runs[0][-1]["summary_kind"] == "document_overview"


async def test_scheduled_network_failure_is_logged_and_preserves_error_reason(
    caplog,
) -> None:
    """A background LLM failure becomes a visible summary state without escaping."""
    repository = SummaryRepository()
    runner = SimpleNamespace(
        run=AsyncMock(side_effect=OSError("LLM endpoint connection refused"))
    )
    document, ingestion = _document()
    service = DocumentSummaryService(repository, runner)  # type: ignore[arg-type]

    scheduled = await service.schedule_summary("project", document, ingestion)
    await service.close()

    assert scheduled.status == SummaryStatus.GENERATING
    assert repository.summary.status == SummaryStatus.FAILED
    assert repository.summary.error == "LLM endpoint connection refused"
    assert "Document summary generation failed" in caplog.text


async def test_scheduled_task_logs_unexpected_exceptions(caplog, monkeypatch) -> None:
    """Unexpected background exceptions are retrieved instead of being unobserved."""
    repository = SummaryRepository()
    runner = SimpleNamespace(run=AsyncMock())
    document, ingestion = _document()
    service = DocumentSummaryService(repository, runner)  # type: ignore[arg-type]

    async def raise_unexpected(*args: object):
        raise RuntimeError("unexpected task failure")

    monkeypatch.setattr(service, "_generate_summary", raise_unexpected)

    await service.schedule_summary("project", document, ingestion)
    await service.close()

    assert not service._tasks
    assert "Unexpected document summary task failure" in caplog.text


async def test_scheduled_summary_exposes_generating_state_before_completion() -> None:
    """Post-ingestion scheduling persists lifecycle state without delaying upload."""
    repository = SummaryRepository()
    payload = GroundedResponsePayload(
        turn_title="Document overview",
        answer_text="Supported overview.",
        claims=[{"text": "Supported overview.", "source_chunk_ids": ["c1"]}],
        insufficient_evidence=False,
    )
    runner = SimpleNamespace(
        run=AsyncMock(
            return_value=(
                "run",
                Answer("Rendered", payload=payload.model_dump()),
                {"status": "completed", "answer_payload": payload.model_dump()},
            )
        )
    )
    document, ingestion = _document()
    service = DocumentSummaryService(repository, runner)  # type: ignore[arg-type]

    scheduled = await service.schedule_summary("project", document, ingestion)

    assert scheduled.status == SummaryStatus.GENERATING
    await service.close()
    assert repository.summary.status == SummaryStatus.READY
