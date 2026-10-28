"""Project and document commands for the persistent CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_grounding.application.bootstrap import PhaseOneApplication
from llm_grounding.application.files import validate_pdf_path
from llm_grounding.cli.formatting import print_json, print_table


def register(
    parser: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register project and document subcommands."""
    project = parser.add_parser("project", help="Manage persistent project corpora")
    commands = project.add_subparsers(dest="project_command", required=True)
    create = commands.add_parser("create", help="Create a project")
    create.add_argument("name")
    create.add_argument(
        "--ingestion-pipeline-id",
        help="Select an active ingestion pipeline for future documents",
    )
    commands.add_parser("list", help="List projects")
    inspect = commands.add_parser("inspect", help="Inspect one project")
    inspect.add_argument("project_id")
    delete = commands.add_parser("delete", help="Delete a project")
    delete.add_argument("project_id")
    add = commands.add_parser("add-document", help="Attach and ingest a PDF")
    add.add_argument("project_id")
    add.add_argument("pdf_file", type=Path)
    add.add_argument(
        "--ingestion", type=Path, help="Ingestion YAML (default: selected pipeline)"
    )
    set_pipeline = commands.add_parser(
        "set-ingestion-pipeline",
        help="Select the ingestion pipeline used by future documents",
    )
    set_pipeline.add_argument("project_id")
    set_pipeline.add_argument("pipeline_id")


async def execute(
    args: argparse.Namespace, application: PhaseOneApplication
) -> int:
    """Execute one project command through ``ProjectService``."""
    command = args.project_command
    if command == "create":
        project = await application.projects.create_project(
            args.name, args.ingestion_pipeline_id
        )
        print(f"Created project {project.id}: {project.name}")
    elif command == "list":
        print_table(
            ["ID", "NAME"],
            [
                [project.id, project.name]
                for project in await application.projects.list_projects()
            ],
        )
    elif command == "inspect":
        print_json(await application.projects.get_project(args.project_id))
    elif command == "delete":
        await application.projects.delete_project(args.project_id)
        print(f"Deleted project {args.project_id}")
    elif command == "add-document":
        ingestion = application.ingestion_config
        if ingestion is None:
            raise RuntimeError("Ingestion configuration is unavailable")
        document, representation = await application.projects.add_document(
            args.project_id,
            validate_pdf_path(args.pdf_file),
            ingestion,
        )
        print(
            f"Document {document.id} ingested as {representation.id} "
            f"({representation.status})."
        )
    elif command == "set-ingestion-pipeline":
        project = await application.projects.set_ingestion_pipeline_for_future(
            args.project_id, args.pipeline_id
        )
        print(
            f"Project {project.id} will use ingestion pipeline "
            f"{project.ingestion_pipeline_id} for future documents."
        )
    else:
        raise ValueError(f"Unsupported project command: {command}")
    return 0
