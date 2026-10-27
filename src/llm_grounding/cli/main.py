"""Argument parsing and dispatch for the ``llm-grounding`` CLI."""

import argparse
import asyncio
import sys

from llm_grounding.cli import chats, components, models, pipelines, projects, vector_dbs
from llm_grounding.cli.context import start_application
from llm_grounding.infrastructure.config.env import load_environment


def create_phase_one_parser() -> argparse.ArgumentParser:
    """Create the persistent workspace and discovery command parser."""
    parser = argparse.ArgumentParser(description="LLM Grounding Project commands")
    commands = parser.add_subparsers(dest="command", required=True)
    projects.register(commands)
    chats.register(commands)
    pipelines.register(commands)
    components.register(commands)
    models.register(commands)
    vector_dbs.register(commands)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Dispatch one CLI command with concise, nonzero-error handling."""
    load_environment()
    parser = create_phase_one_parser()
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(arguments)
    try:
        return asyncio.run(_run_command(args))
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted.\n")


async def _run_command(args: argparse.Namespace) -> int:
    """Route commands to thin domain-specific CLI modules."""
    if args.command == "component":
        return components.execute(args)
    if args.command == "model":
        return await models.execute(args)
    if args.command == "vector-db":
        return await vector_dbs.execute(args)
    if args.command == "pipeline" and args.pipeline_command == "validate":
        return pipelines.validate(args)

    application = await start_application(args)
    try:
        if args.command == "project":
            return await projects.execute(args, application)
        if args.command == "chat":
            return await chats.execute(args, application)
        if args.command == "pipeline":
            return await pipelines.execute(args, application)
        raise ValueError(f"Unsupported command: {args.command}")
    finally:
        await application.close()
