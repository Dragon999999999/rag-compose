"""Persisted pipeline management commands."""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_grounding.application.bootstrap import PhaseOneApplication
from llm_grounding.cli.formatting import print_json, print_table
from llm_grounding.infrastructure.config.loader import load_pipeline_config


def register(
    parser: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register pipeline management commands."""
    pipeline = parser.add_parser("pipeline", help="Manage persisted RAG pipelines")
    commands = pipeline.add_subparsers(dest="pipeline_command", required=True)
    list_command = commands.add_parser("list", help="List persisted pipelines")
    list_command.add_argument(
        "--type",
        dest="pipeline_type",
        choices=("ingestion", "query"),
        help="Restrict results to one pipeline type",
    )
    inspect = commands.add_parser("inspect", help="Inspect one persisted pipeline")
    inspect.add_argument("pipeline_id")
    validate = commands.add_parser(
        "validate", help="Validate a pipeline YAML without importing it"
    )
    validate.add_argument("config_file", type=Path)
    create = commands.add_parser(
        "create", help="Create an API-owned pipeline from YAML"
    )
    create.add_argument("config_file", type=Path)
    pipeline_import = commands.add_parser(
        "import", help="Synchronize pipeline YAML files into PostgreSQL"
    )
    pipeline_import.add_argument(
        "directory",
        type=Path,
        default=None,
        nargs="?",
        help="Pipeline directory (default: canonical config/pipelines)",
    )
    deactivate = commands.add_parser(
        "delete", help="Deactivate one persisted pipeline"
    )
    deactivate.add_argument("pipeline_id")
    repair = commands.add_parser(
        "repair-representation-hashes",
        help="Repair verified legacy ingestion hashes without changing configurations",
    )
    repair.add_argument(
        "--dry-run",
        action="store_true",
        help="Report eligible repairs without writing metadata",
    )


def validate(args: argparse.Namespace) -> int:
    """Validate a pipeline file without starting database-backed services."""
    config = load_pipeline_config(args.config_file)
    print_table(
        ["STATUS", "TYPE", "NAME", "CONFIG_HASH"],
        [["valid", config.kind, config.name, config.config_hash]],
    )
    return 0


async def execute(
    args: argparse.Namespace, application: PhaseOneApplication
) -> int:
    """Execute one persisted pipeline command through ``PipelineService``."""
    if application.pipeline_service is None:
        raise RuntimeError("Pipeline management is unavailable")
    command = args.pipeline_command
    if command == "list":
        records = await application.pipeline_service.list(kind=args.pipeline_type)
        print_table(
            ["ID", "TYPE", "NAME", "VERSION", "SOURCE", "ACTIVE"],
            [
                [
                    record.id,
                    record.kind,
                    record.name,
                    record.version,
                    record.source_kind,
                    record.active,
                ]
                for record in records
            ],
        )
    elif command == "inspect":
        print_json(await application.pipeline_service.get(args.pipeline_id))
    elif command == "create":
        record = await application.pipeline_service.create(
            load_pipeline_config(args.config_file)
        )
        print(f"Created pipeline {record.id}: {record.name}")
    elif command == "import":
        records = await application.pipeline_service.list()
        print_table(
            ["ID", "TYPE", "NAME", "VERSION"],
            [
                [record.id, record.kind, record.name, record.version]
                for record in records
            ],
        )
    elif command == "delete":
        await application.pipeline_service.deactivate(args.pipeline_id)
        print(f"Deactivated pipeline {args.pipeline_id}")
    elif command == "repair-representation-hashes":
        result = await application.pipeline_service.repair_ingestion_hashes(
            dry_run=args.dry_run
        )
        print_json(result.as_dict())
    else:
        raise ValueError(f"Unsupported pipeline command: {command}")
    return 0
