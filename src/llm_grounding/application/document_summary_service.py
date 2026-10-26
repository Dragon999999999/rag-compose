"""Persisted grounded summaries generated after successful PDF ingestion."""

import asyncio
import logging
from datetime import UTC, datetime

from llm_grounding.domain.corpus import (
    CorpusRequest,
    Document,
    DocumentIngestion,
    DocumentSummary,
    SummaryStatus,
    TaskType,
)
from llm_grounding.infrastructure.database.repositories import PostgresRepository
from llm_grounding.orchestration.basic_rag import BasicRagRunner

DOCUMENT_SUMMARY_TASK = (
    "Create a concise, evidence-grounded overview of this document. Cover its "
    "purpose, main approach or content, key findings or conclusions where "
    "supported, and notable limitations when available."
)

_LOGGER = logging.getLogger(__name__)


class DocumentSummaryService:
    """Generate and retain one verified document overview per ingestion version."""

    def __init__(self, repository: PostgresRepository, runner: BasicRagRunner) -> None:
        """Inject persistence and the established grounded summary workflow."""
        self._repository = repository
        self._runner = runner
        self._tasks: dict[tuple[str, str], asyncio.Task[DocumentSummary]] = {}
        self._scheduling_lock = asyncio.Lock()

    async def schedule_summary(
        self,
        project_id: str,
        document: Document,
        ingestion: DocumentIngestion,
    ) -> DocumentSummary:
        """Start a persisted summary in the background after ingestion is ready."""
        async with self._scheduling_lock:
            summary, should_generate = await self._begin_if_needed(document, ingestion)
            self._schedule_generation_if_needed(
                project_id,
                document,
                ingestion,
                summary,
                should_generate,
            )
            return summary

    async def regenerate_summary(
        self,
        project_id: str,
        document: Document,
        ingestion: DocumentIngestion,
    ) -> DocumentSummary:
        """Explicitly regenerate a document overview without replacing active work.

        A ready or failed summary is reset and generated again. The initiating
        caller awaits its new task so it receives the completed state without
        polling. Repeated requests during an active generation return the current
        lifecycle record and leave the existing task untouched.
        """
        key = (document.id, ingestion.id)
        task: asyncio.Task[DocumentSummary] | None = None
        async with self._scheduling_lock:
            existing = await self._repository.document_summary(*key)
            if existing is not None and existing.status == SummaryStatus.GENERATING:
                if key in self._tasks:
                    return existing

            page_count = await self._repository.page_count_for_ingestion(ingestion.id)
            summary = await self._repository.begin_document_summary(
                document.id,
                ingestion.id,
                page_count,
            )
            task = self._schedule_generation_if_needed(
                project_id,
                document,
                ingestion,
                summary,
                True,
            )
        assert task is not None
        # Do not cancel the persisted background task if the HTTP client disconnects.
        return await asyncio.shield(task)

    async def ensure_summary(
        self,
        project_id: str,
        document: Document,
        ingestion: DocumentIngestion,
    ) -> DocumentSummary:
        """Return a stored summary or generate it through the summary retrieval path.

        The source PDF remains usable when generation fails. Failure is persisted for
        the document workspace instead of being surfaced as an ingestion failure.
        """
        summary, should_generate = await self._begin_if_needed(document, ingestion)
        if not should_generate:
            return summary
        return await self._generate_summary(project_id, document, ingestion)

    async def close(self) -> None:
        """Allow in-flight summaries to persist their final lifecycle state."""
        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)

    async def _begin_if_needed(
        self, document: Document, ingestion: DocumentIngestion
    ) -> tuple[DocumentSummary, bool]:
        """Return the existing lifecycle state or persist a new generating state."""
        existing = await self._repository.document_summary(document.id, ingestion.id)
        if existing is not None and existing.status in {
            SummaryStatus.READY,
            SummaryStatus.GENERATING,
        }:
            return existing, False
        page_count = await self._repository.page_count_for_ingestion(ingestion.id)
        return (
            await self._repository.begin_document_summary(
                document.id, ingestion.id, page_count
            ),
            True,
        )

    async def _generate_summary(
        self,
        project_id: str,
        document: Document,
        ingestion: DocumentIngestion,
    ) -> DocumentSummary:
        """Execute and persist the established grounded summary workflow."""
        try:
            run_id, answer, metadata = await self._runner.run(
                CorpusRequest(
                    project_id,
                    DOCUMENT_SUMMARY_TASK,
                    TaskType.SUMMARY,
                    (document.id,),
                )
            )
            if not answer.payload or str(metadata.get("status")) == "rejected":
                raise RuntimeError("Grounded summary validation failed")
            metadata = {
                **metadata,
                "document_id": document.id,
                "ingestion_id": ingestion.id,
                "summary_kind": "document_overview",
            }
            await self._repository.record_run(
                run_id,
                project_id,
                "query_basic_v1",
                1,
                str(metadata["status"]),
                metadata,
            )
            return await self._repository.complete_document_summary(
                document.id, ingestion.id, run_id
            )
        except Exception as exc:
            return await self._persist_failure(document, ingestion, exc)

    def _schedule_generation_if_needed(
        self,
        project_id: str,
        document: Document,
        ingestion: DocumentIngestion,
        summary: DocumentSummary,
        should_generate: bool,
    ) -> asyncio.Task[DocumentSummary] | None:
        """Create one tracked task when the persisted lifecycle requires it."""
        key = (document.id, ingestion.id)
        if not should_generate and (
            summary.status != SummaryStatus.GENERATING or key in self._tasks
        ):
            return None
        task = asyncio.create_task(
            self._generate_summary(project_id, document, ingestion)
        )
        self._tasks[key] = task
        task.add_done_callback(
            lambda completed_task: self._finish_scheduled_task(key, completed_task)
        )
        return task

    async def _persist_failure(
        self,
        document: Document,
        ingestion: DocumentIngestion,
        error: Exception,
    ) -> DocumentSummary:
        """Persist an LLM failure without allowing background work to escape.

        The original exception text is retained for the document workspace. If the
        failure itself cannot be persisted, return an ephemeral failed state and log
        the persistence problem so the scheduled task still completes normally.
        """
        failure_reason = str(error) or type(error).__name__
        _LOGGER.exception(
            "Document summary generation failed for document %s ingestion %s",
            document.id,
            ingestion.id,
        )
        try:
            return await self._repository.fail_document_summary(
                document.id, ingestion.id, failure_reason
            )
        except Exception:
            _LOGGER.exception(
                "Unable to persist summary failure for document %s ingestion %s",
                document.id,
                ingestion.id,
            )
            now = datetime.now(UTC)
            return DocumentSummary(
                document.id,
                ingestion.id,
                SummaryStatus.FAILED,
                None,
                failure_reason,
                None,
                now,
                now,
            )

    def _finish_scheduled_task(
        self,
        key: tuple[str, str],
        task: asyncio.Task[DocumentSummary],
    ) -> None:
        """Remove a completed task and retrieve/log unexpected task failures."""
        if self._tasks.get(key) is task:
            self._tasks.pop(key, None)
        if task.cancelled():
            _LOGGER.info("Document summary task cancelled for document %s", key[0])
            return
        try:
            task.result()
        except Exception:
            _LOGGER.exception(
                "Unexpected document summary task failure for document %s ingestion %s",
                key[0],
                key[1],
            )
