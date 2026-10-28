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
    normalized = requested.casefold()
    return normalized in {
        definition.stage.casefold(),
        definition.location.casefold(),
        getattr(definition, "backend", "").casefold(),
        (definition.provider or "").casefold(),
    }


def _definition_payload(definition: ModelDefinition) -> dict[str, object]:
    """Return model configuration without resolved credentials."""
    return {
        "id": definition.id,
        "stage": definition.stage,
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
                            definition.stage,
                            definition.location,
                            definition.backend,
                            status["availability"],
                        ]
                    )
            print_table(
                ["ID", "STAGE", "LOCATION", "BACKEND", "AVAILABILITY"], rows
            )
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
