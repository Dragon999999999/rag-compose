"""Chat, pipeline-selection, and chat-override CLI commands."""

from __future__ import annotations

import argparse

from llm_grounding.application.bootstrap import PhaseOneApplication
from llm_grounding.cli.formatting import parse_json_value, print_json
from llm_grounding.domain.corpus import TaskType


def register(
    parser: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register project-scoped chat commands and override operations."""
    chat = parser.add_parser("chat", help="Manage project-scoped conversations")
    commands = chat.add_subparsers(dest="chat_command", required=True)
    create = commands.add_parser("create", help="Create a chat")
    create.add_argument("project_id")
    create.add_argument("name")
    create.add_argument("--pipeline-id", help="Select an active query pipeline")
    commands.add_parser("list", help="List chats across projects").add_argument(
        "--project-id", help="Restrict the list to one project"
    )
    inspect = commands.add_parser("inspect", help="Inspect one chat")
    inspect.add_argument("chat_id")
    set_pipeline = commands.add_parser(
        "set-pipeline", help="Select an active query pipeline for a chat"
    )
    set_pipeline.add_argument("chat_id")
    set_pipeline.add_argument("pipeline_id")
    ask = commands.add_parser("ask", help="Ask a grounded project question")
    ask.add_argument("chat_id")
    ask.add_argument("message")
    ask.add_argument(
        "--summary", action="store_true", help="Use summary retrieval planning"
    )
    ask.add_argument(
        "--pipeline-id", help="Select a persisted pipeline runtime for this turn"
    )
    ask.add_argument(
        "--grounding-diagnostics",
        choices=("off", "summary", "verbose"),
        default="off",
        help="Emit grounding diagnostics to stderr (default: off)",
    )
    override = commands.add_parser("override", help="Manage persisted chat overrides")
    override_commands = override.add_subparsers(
        dest="override_command", required=True
    )
    list_overrides = override_commands.add_parser(
        "list", help="List overrides for a chat"
    )
    list_overrides.add_argument("chat_id")
    set_override = override_commands.add_parser("set", help="Set one stage override")
    set_override.add_argument("chat_id")
    set_override.add_argument("stage")
    set_override.add_argument("parameter")
    set_override.add_argument("value", help="Value encoded as JSON")
    remove = override_commands.add_parser(
        "remove", help="Remove one stage override"
    )
    remove.add_argument("chat_id")
    remove.add_argument("stage")
    remove.add_argument("parameter")
    reset = override_commands.add_parser("reset", help="Reset chat overrides")
    reset.add_argument("chat_id")
    reset.add_argument("stage", nargs="?", help="Reset only one stage")


async def execute(
    args: argparse.Namespace, application: PhaseOneApplication
) -> int:
    """Execute one chat command through ``ChatService``."""
    command = args.chat_command
    if command == "create":
        chat = await application.chats.create_chat(
            args.project_id, args.name, args.pipeline_id
        )
        print(f"Created chat {chat.id}: {chat.name}")
    elif command == "list":
        chats = (
            await application.chats.list_chats(args.project_id)
            if args.project_id
            else await application.chats.list_all_chats()
        )
        for chat in chats:
            print(
                f"{chat.id}\t{chat.project_id}\t{chat.name}\t"
                f"{chat.pipeline_id or '-'}"
            )
    elif command == "inspect":
        print_json(await application.chats.get_chat(args.chat_id))
    elif command == "set-pipeline":
        chat = await application.chats.set_pipeline(args.chat_id, args.pipeline_id)
        print(f"Chat {chat.id} now uses query pipeline {chat.pipeline_id}.")
    elif command == "ask":
        task_type = TaskType.SUMMARY if args.summary else TaskType.QA
        answer = await application.chats.ask(
            args.chat_id, args.message, task_type, args.pipeline_id
        )
        print(answer.text)
    elif command == "override":
        await _execute_override(args, application)
    else:
        raise ValueError(f"Unsupported chat command: {command}")
    return 0


async def _execute_override(
    args: argparse.Namespace, application: PhaseOneApplication
) -> None:
    """Execute one override mutation using the validated chat service boundary."""
    command = args.override_command
    if command == "list":
        print_json((await application.chats.get_chat(args.chat_id)).stage_overrides)
    elif command == "set":
        chat = await application.chats.set_stage_override(
            args.chat_id,
            args.stage,
            args.parameter,
            parse_json_value(args.value),
        )
        print_json(chat.stage_overrides)
    elif command == "remove":
        chat = await application.chats.remove_stage_override(
            args.chat_id, args.stage, args.parameter
        )
        print_json(chat.stage_overrides)
    elif command == "reset":
        chat = await application.chats.reset_stage_overrides(args.chat_id, args.stage)
        print_json(chat.stage_overrides)
    else:
        raise ValueError(f"Unsupported override command: {command}")
