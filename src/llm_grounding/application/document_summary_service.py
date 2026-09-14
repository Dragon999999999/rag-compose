"""Persisted grounded summaries generated after successful PDF ingestion."""

import asyncio

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


class DocumentSummaryService:
    """Generate and retain one verified document overview per ingestion version."""

    def __init__(
        self, repository: PostgresRepository, runner: BasicRagRunner
    ) -> None:
        """Inject persistence and the established grounded summary workflow."""
        self._repository = repository
        self._runner = runner
        self._tasks: set[asyncio.Task[DocumentSummary]] = set()

    async def schedule_summary(
        self,
        project_id: str,
        document: Document,
        ingestion: DocumentIngestion,
    ) -> DocumentSummary:
        """Start a persisted summary in the background after ingestion is ready."""
        summary, should_generate = await self._begin_if_needed(document, ingestion)
        if should_generate:
            task = asyncio.create_task(
                self._generate_summary(project_id, document, ingestion)
            )
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        return summary

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
            await asyncio.gather(*self._tasks, return_exceptions=True)

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
                "basic_rag_v1",
                1,
                str(metadata["status"]),
                metadata,
            )
            return await self._repository.complete_document_summary(
                document.id, ingestion.id, run_id
            )
        except (RuntimeError, ValueError) as exc:
            return await self._repository.fail_document_summary(
                document.id, ingestion.id, type(exc).__name__
            )
