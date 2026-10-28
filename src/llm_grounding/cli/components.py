"""Configuration and runtime component discovery commands."""

from __future__ import annotations

import argparse
from importlib.resources import files
from pathlib import Path

import yaml

from llm_grounding.cli.formatting import print_json, print_table
from llm_grounding.infrastructure.config.declarative import ComponentDefinition
from llm_grounding.infrastructure.config.loader import load_configuration_registry
from llm_grounding.orchestration.registry import ComponentRegistry

_STAGE_ALIASES = {
    "parser": "parser",
    "parsers": "parser",
    "chunker": "chunker",
    "chunkers": "chunker",
    "embedder": "embedding",
    "embedders": "embedding",
    "embedding": "embedding",
    "embeddings": "embedding",
    "vector-db": "vector_backend",
    "vector-dbs": "vector_backend",
    "vector_backend": "vector_backend",
    "retriever": "retriever",
    "retrievers": "retriever",
    "reranker": "reranker",
    "rerankers": "reranker",
    "generator": "reasoner",
    "generators": "reasoner",
    "reasoner": "reasoner",
    "reasoners": "reasoner",
    "validator": "assessor",
    "validators": "assessor",
    "assessor": "assessor",
    "assessors": "assessor",
    "knowledge-builder": "knowledge_builder",
    "knowledge_builder": "knowledge_builder",
    "decision-policy": "decision_policy",
    "decision_policy": "decision_policy",
    "renderer": "renderer",
}


def register(
    parser: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register component discovery commands."""
    component = parser.add_parser(
        "component", help="Discover configured and registered components"
    )
    commands = component.add_subparsers(dest="component_command", required=True)
    list_command = commands.add_parser("list", help="List component definitions")
    list_command.add_argument(
        "--stage",
        choices=tuple(sorted(_STAGE_ALIASES)),
        help="Filter by component family or a common stage alias",
    )
    inspect = commands.add_parser("inspect", help="Inspect one component definition")
    inspect.add_argument("component")


def _schema_path(category: str) -> Path:
    """Resolve a component-family schema in a checkout or installed package."""
    checkout = Path(__file__).resolve().parents[3] / "config/components" / category
    if (checkout / "schema.yaml").is_file():
        return checkout / "schema.yaml"
    packaged = (
        files("llm_grounding.infrastructure.config")
        / "standalone/components"
        / category
        / "schema.yaml"
    )
    return Path(str(packaged))


def _schema(category: str) -> dict[str, object]:
    """Load the family schema that documents parameters and their defaults."""
    path = _schema_path(category)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _runtime_implementations() -> dict[str, list[str]]:
    """Expose implementation names from the executable component registry."""
    registry = ComponentRegistry()
    return {
        "parser": sorted(registry.parsers),
        "knowledge_builder": sorted(registry.knowledge_builders),
        "chunker": sorted(registry.chunkers),
        "retriever": sorted(registry.retrievers),
        "reranker": sorted(registry.rerankers),
        "reasoner": sorted(registry.pipeline_reasoners),
        "assessor": sorted(registry.pipeline_assessors),
        "decision_policy": sorted(registry.pipeline_decision_policies),
        "renderer": sorted(registry.pipeline_renderers),
        "vector_backend": sorted(registry.vector_backends),
    }


def _definition_payload(
    category: str, definition: ComponentDefinition
) -> dict[str, object]:
    """Build one discovery record with configured values and schema defaults."""
    return {
        "id": definition.id,
        "stage": category,
        "type": definition.type,
        "implementation": definition.implementation,
        "description": definition.description,
        "configured": definition.config,
        "requirements": (
            definition.requirements.model_dump(mode="json")
            if definition.requirements is not None
            else {}
        ),
        "schema": _schema(category),
    }


def execute(args: argparse.Namespace) -> int:
    """Execute a read-only component discovery command."""
    registry = load_configuration_registry()
    if args.component_command == "list":
        category = _STAGE_ALIASES.get(args.stage) if args.stage else None
        rows: list[list[object]] = []
        for component_id, (actual_category, definition) in sorted(
            registry.components.items()
        ):
            if category is not None and actual_category != category:
                continue
            rows.append(
                [
                    component_id,
                    actual_category,
                    definition.implementation,
                    definition.description,
                ]
            )
        print_table(["ID", "STAGE", "IMPLEMENTATION", "DESCRIPTION"], rows)
        return 0
    if args.component_command == "inspect":
        try:
            category, definition = registry.components[args.component]
        except KeyError as exc:
            raise ValueError(
                f"Unknown component configuration ID: {args.component}"
            ) from exc
        payload = _definition_payload(category, definition)
        payload["registered_implementations"] = _runtime_implementations().get(
            category, []
        )
        print_json(payload)
        return 0
    raise ValueError(f"Unsupported component command: {args.component_command}")
