"""Terminal commands for Baseline 0 and persistent Basic RAG v1 workspaces."""

import argparse
import asyncio
import sys
from pathlib import Path

from llm_grounding.application.bootstrap import create_phase_one_application
from llm_grounding.application.service import RagService, validate_pdf_path
from llm_grounding.domain.corpus import TaskType
from llm_grounding.infrastructure.config.loader import (
    load_architecture_config,
    load_ingestion_config,
)
from llm_grounding.orchestration.runner import RunFailedError


def create_parser() -> argparse.ArgumentParser:
    """Create the unchanged Baseline 0 command parser."""
    parser = argparse.ArgumentParser(description="Basic Summary Extraction Baseline")
    parser.add_argument("pdf_file", type=Path, help="Research paper PDF to process")
    parser.add_argument(
        "--architecture",
        type=Path,
        help="Architecture YAML (default: packaged baseline_v0)",
    )
    return parser


def create_phase_one_parser() -> argparse.ArgumentParser:
    """Create persistent project and chat commands without changing Baseline syntax."""
    parser = argparse.ArgumentParser(
        description="LLM Grounding Project corpus commands"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    project = commands.add_parser("project", help="Manage persistent project corpora")
    project_commands = project.add_subparsers(dest="project_command", required=True)
    create = project_commands.add_parser("create", help="Create a project")
    create.add_argument("name")
    project_commands.add_parser("list", help="List projects")
    add = project_commands.add_parser("add-document", help="Attach and ingest a PDF")
    add.add_argument("project_id")
    add.add_argument("pdf_file", type=Path)
    add.add_argument(
        "--ingestion", type=Path, help="Ingestion YAML (default: packaged config)"
    )
    chat = commands.add_parser("chat", help="Manage project-scoped conversations")
    chat_commands = chat.add_subparsers(dest="chat_command", required=True)
    create_chat = chat_commands.add_parser("create", help="Create a chat")
    create_chat.add_argument("project_id")
    create_chat.add_argument("name")
    ask = chat_commands.add_parser("ask", help="Ask a grounded project question")
    ask.add_argument("chat_id")
    ask.add_argument("message")
    ask.add_argument(
        "--summary", action="store_true", help="Use summary retrieval planning"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run a baseline extraction or dispatch a persistent Basic RAG command."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    phase_one = bool(arguments) and arguments[0] in {"project", "chat"}
    parser = create_phase_one_parser() if phase_one else create_parser()
    args = parser.parse_args(arguments)
    try:
        if phase_one:
            return asyncio.run(_run_phase_one_command(args))
        path = validate_pdf_path(args.pdf_file)
        config = load_architecture_config(args.architecture)
        service = RagService(config)
        run = asyncio.run(service.process_pdf(path))
    except (OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    except RunFailedError as exc:
        parser.exit(
            1,
            f"error: {exc} Check PDF content, architecture settings, and "
            "AQUEDUCT_API_KEY or AQUEDUCT_API_KEY_FILE; model requests require "
            "a reachable endpoint and valid structured output.\n",
        )
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted.\n")
    if run.answer is None:
        parser.exit(1, "error: Completed run did not produce an answer.\n")
    print(run.answer.text)
    print("\nExtraction successful.")
    return 0


async def _run_phase_one_command(args: argparse.Namespace) -> int:
    """Dispatch thin application commands while preserving composition in YAML."""
    architecture = load_architecture_config(
        Path("configs/architectures/basic_rag_v1.yaml")
    )
    ingestion = load_ingestion_config(getattr(args, "ingestion", None))
    application = create_phase_one_application(architecture, ingestion)
    try:
        if args.command == "project" and args.project_command == "create":
            project = await application.projects.create_project(args.name)
            print(f"Created project {project.id}: {project.name}")
        elif args.command == "project" and args.project_command == "list":
            for project in await application.projects.list_projects():
                print(f"{project.id}\t{project.name}")
        elif args.command == "project" and args.project_command == "add-document":
            document, representation = await application.projects.add_document(
                args.project_id,
                validate_pdf_path(args.pdf_file),
                ingestion,
            )
            print(
                "Document "
                f"{document.id} ingested as {representation.id} ({representation.status})."
            )
        elif args.command == "chat" and args.chat_command == "create":
            chat = await application.chats.create_chat(args.project_id, args.name)
            print(f"Created chat {chat.id}: {chat.name}")
        elif args.command == "chat" and args.chat_command == "ask":
            task_type = TaskType.SUMMARY if args.summary else TaskType.QA
            answer = await application.chats.ask(args.chat_id, args.message, task_type)
            print(answer.text)
        else:
            raise ValueError("Unsupported command")
    finally:
        await application.close()
    return 0
