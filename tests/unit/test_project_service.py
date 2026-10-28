"""Test project application-service operations exposed through the browser API."""

from datetime import UTC, datetime

from llm_grounding.application.project_service import ProjectService
from llm_grounding.domain.corpus import Project


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
