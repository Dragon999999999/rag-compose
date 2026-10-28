"""Verify the reorganized CLI parser and discovery adapters."""

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from llm_grounding.application.project_service import ProjectService
from llm_grounding.cli.chats import execute as execute_chats
from llm_grounding.cli.components import execute as execute_components
from llm_grounding.cli.formatting import print_table
from llm_grounding.cli.main import create_phase_one_parser
from llm_grounding.domain.corpus import Chat, Pipeline, Project


def test_cli_parser_preserves_existing_commands_and_adds_discovery_commands() -> None:
    """The parser exposes legacy commands and all requested command groups."""
    parser = create_phase_one_parser()

    project = parser.parse_args(["project", "set-ingestion-pipeline", "p", "i"])
    pipeline = parser.parse_args(["pipeline", "list", "--type", "query"])
    repair = parser.parse_args(
        ["pipeline", "repair-representation-hashes", "--dry-run"]
    )
    chat = parser.parse_args(
        ["chat", "override", "set", "c", "retriever", "top_k", "10"]
    )
    component = parser.parse_args(["component", "list", "--stage", "retrievers"])
    model = parser.parse_args(["model", "inspect", "embedding_001"])
    vector_db = parser.parse_args(["vector-db", "test", "qdrant"])

    assert (project.command, project.project_command) == (
        "project",
        "set-ingestion-pipeline",
    )
    assert pipeline.pipeline_type == "query"
    assert repair.dry_run
    assert chat.override_command == "set"
    assert component.stage == "retrievers"
    assert model.model_command == "inspect"
    assert vector_db.vector_db_command == "test"


def test_component_list_filters_configured_stage(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Component discovery uses the standalone registry and stage filter."""
    parser = create_phase_one_parser()
    args = parser.parse_args(["component", "list", "--stage", "retrievers"])

    assert execute_components(args) == 0
    output = capsys.readouterr().out

    assert "retriever_dense_vector_default" in output
    assert "retriever" in output
    assert "parser_pymupdf" not in output


def test_print_table_aligns_columns_to_longest_value(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Table columns remain aligned when cells have different lengths."""
    print_table(
        ["ID", "NAME", "STATUS"],
        [["a", "short", "ready"], ["long-id", "longer name", "-"]],
    )

    assert capsys.readouterr().out == (
        "ID       NAME         STATUS\n"
        "a        short        ready\n"
        "long-id  longer name  -\n"
    )


@pytest.mark.asyncio
async def test_set_ingestion_pipeline_for_future_documents_does_not_touch_documents(
) -> None:
    """Project pipeline selection delegates only to configuration persistence."""
    now = datetime.now(UTC)
    project = Project("project", "Corpus", now)
    pipeline = Pipeline(
        "ingestion",
        "ingestion_default_v1",
        1,
        1,
        "",
        {},
        "config-hash",
        "representation-hash",
        "yaml",
        None,
        None,
        True,
        True,
        now,
        now,
        "ingestion",
    )

    class Repository:
        async def set_project_ingestion_pipeline(
            self, project_id: str, pipeline_id: str, **kwargs
        ):
            assert project_id == project.id
            assert pipeline_id == pipeline.id
            assert kwargs["representation_hash"] == pipeline.representation_hash
            return project

    service = ProjectService(
        Repository(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        pipeline_service=SimpleNamespace(
            select_ingestion=lambda pipeline_id: _return(pipeline),
            resolve_ingestion=lambda pipeline_id: _return(object()),
        ),
    )

    selected = await service.set_ingestion_pipeline_for_future(project.id, pipeline.id)

    assert selected == project


@pytest.mark.asyncio
async def test_chat_override_set_uses_chat_service_validation_boundary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI passes parsed override values to the shared chat service."""
    parser = create_phase_one_parser()
    args = parser.parse_args(
        ["chat", "override", "set", "chat", "retriever", "top_k", "10"]
    )
    captured: list[tuple[str, str, str, object]] = []
    chat = Chat("chat", "project", "Chat", datetime.now(UTC))

    class Chats:
        async def set_stage_override(
            self, chat_id: str, stage: str, parameter: str, value: object
        ) -> Chat:
            captured.append((chat_id, stage, parameter, value))
            return chat

    application = SimpleNamespace(chats=Chats())

    assert await execute_chats(args, application) == 0
    assert captured == [("chat", "retriever", "top_k", 10)]
    assert '"retriever"' not in capsys.readouterr().out


async def _return(value):
    """Return a value from a small async fake service method."""
    return value
