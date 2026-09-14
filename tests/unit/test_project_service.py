"""Test project application-service operations exposed through the browser API."""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

from llm_grounding.application.project_service import ProjectService
from llm_grounding.domain.corpus import (
    Document,
    DocumentIngestion,
    IngestionStatus,
    Project,
)


class RecordingRepository:
    """Record project rename requests without database infrastructure."""

    def __init__(self) -> None:
        """Initialize the next canonical project result."""
        self.calls: list[tuple[str, str]] = []

    async def rename_project(self, project_id: str, name: str) -> Project:
        """Record the application-level rename request."""
        self.calls.append((project_id, name))
        return Project(project_id, name, datetime.now(UTC))


async def test_rename_project_uses_repository_boundary() -> None:
    """The service delegates renaming without HTTP or SQL behavior."""
    repository = RecordingRepository()
    service = ProjectService(repository, object(), object())  # type: ignore[arg-type]

    project = await service.rename_project("project-id", "Renamed project")

    assert project.name == "Renamed project"
    assert repository.calls == [("project-id", "Renamed project")]


async def test_add_document_starts_grounded_summary_after_successful_ingestion(
) -> None:
    """The project workflow starts summary persistence only after ingestion is ready."""
    now = datetime.now(UTC)
    document = Document("document", "paper.pdf", "digest", "pdf/digest.pdf", now)
    ingestion = DocumentIngestion(
        "ingestion",
        document.id,
        "config",
        "{}",
        "embedding",
        8,
        IngestionStatus.READY,
        now,
    )
    ingestion_service = SimpleNamespace(
        add_document=AsyncMock(return_value=(document, object(), ingestion))
    )
    summaries = SimpleNamespace(schedule_summary=AsyncMock())
    service = ProjectService(
        object(), ingestion_service, object(), summaries  # type: ignore[arg-type]
    )

    stored_document, stored_ingestion = await service.add_document(
        "project", __file__, object()  # type: ignore[arg-type]
    )

    assert (stored_document, stored_ingestion) == (document, ingestion)
    summaries.schedule_summary.assert_awaited_once_with("project", document, ingestion)
