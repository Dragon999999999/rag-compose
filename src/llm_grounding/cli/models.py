"""Model registry discovery commands with local runtime availability."""

from __future__ import annotations

import argparse

from llm_grounding.cli.formatting import print_json, print_table
from llm_grounding.infrastructure.config.declarative import ModelDefinition
from llm_grounding.infrastructure.config.loader import load_model_registry
from llm_grounding.infrastructure.model_management.manager import LocalModelManager


def register(
    parser: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register model discovery commands."""
    model = parser.add_parser("model", help="Discover configured models")
    commands = model.add_subparsers(dest="model_command", required=True)
    list_command = commands.add_parser("list", help="List configured models")
    list_command.add_argument(
        "--type",
        dest="model_type",
        help="Filter by stage, backend, location, or provider",
    )
    inspect = commands.add_parser("inspect", help="Inspect one configured model")
    inspect.add_argument("model")
    download = commands.add_parser(
        "download", help="Download and verify local model weights"
    )
    download.add_argument("models", nargs="+", help="Model IDs to prepare")


async def _status(
    manager: LocalModelManager, definition: ModelDefinition
) -> dict[str, object]:
    """Read local runtime state without downloading or starting a model."""
    if definition.location != "local":
        return {"configured": True, "availability": "remote"}
    try:
        status = await manager.status(definition.id)
    except (OSError, RuntimeError, ValueError) as exc:
        return {"configured": True, "availability": "unavailable", "detail": str(exc)}
    return {
        "configured": True,
        "availability": "ready" if status.weights_available else "weights_missing",
        "weights_available": status.weights_available,
        "runtime_type": status.runtime_type,
        "running": status.running,
        "active_leases": status.active_leases,
        "detail": status.detail,
    }


def _matches(definition: ModelDefinition, requested: str | None) -> bool:
    """Match the user-facing model type filter against safe registry fields."""
    if requested is None:
        return True
    normalized = requested.casefold().replace("_", "-")
    if normalized in {"knowledge", "knowledge-building"}:
        normalized = "validation"
    return normalized in {
        definition.stage.casefold().replace("_", "-"),
        definition.location.casefold(),
        getattr(definition, "backend", "").casefold(),
        (definition.provider or "").casefold(),
    }


def _display_stage(definition: ModelDefinition) -> str:
    """Return the shorter CLI label for models used in knowledge building."""
    if definition.stage.casefold() == "validation":
        return "knowledge"
    return definition.stage


def _definition_payload(definition: ModelDefinition) -> dict[str, object]:
    """Return model configuration without resolved credentials."""
    return {
        "id": definition.id,
        "stage": _display_stage(definition),
        "location": definition.location,
        "backend": definition.backend,
        "provider": definition.provider,
        "name": definition.name,
        "connection": definition.connection,
        "source": definition.source,
        "storage": definition.storage,
        "runtime": definition.runtime,
        "parameters": definition.parameters,
    }


async def execute(args: argparse.Namespace) -> int:
    """Execute a read-only model discovery command."""
    registry = load_model_registry()
    manager = LocalModelManager(registry)
    try:
        if args.model_command == "list":
            rows: list[list[object]] = []
            for definition in registry.list():
                if _matches(definition, args.model_type):
                    status = await _status(manager, definition)
                    rows.append(
                        [
                            definition.id,
                            _display_stage(definition),
                            definition.location,
                            definition.backend,
                            status["availability"],
                        ]
                    )
            print_table(
                ["ID", "STAGE", "LOCATION", "BACKEND", "AVAILABILITY"], rows
            )
            return 0
        if args.model_command == "download":
            rows: list[list[object]] = []
            for model_id in args.models:
                definition = registry.resolve(model_id)
                if definition.location != "local":
                    raise ValueError(
                        f"Model {model_id} is remote and has no local weights "
                        "to download"
                    )
                print(f"Preparing {model_id} ({definition.name})...")
                path = await manager.prepare(model_id)
                rows.append([model_id, _display_stage(definition), path, "ready"])
            print_table(["ID", "STAGE", "WEIGHTS", "STATUS"], rows)
            return 0
        if args.model_command == "inspect":
            definition = registry.resolve(args.model)
            payload = _definition_payload(definition)
            payload["availability"] = await _status(manager, definition)
            print_json(payload)
            return 0
        raise ValueError(f"Unsupported model command: {args.model_command}")
    finally:
        await manager.close()
