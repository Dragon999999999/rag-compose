"""Vector-database discovery and connectivity checks."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from llm_grounding.cli.formatting import print_json, print_table
from llm_grounding.infrastructure.config.declarative import ComponentDefinition
from llm_grounding.infrastructure.config.loader import load_configuration_registry
from llm_grounding.infrastructure.config.settings import FaissSettings, QdrantSettings
from llm_grounding.infrastructure.vector.factory import VectorStoreFactory


def register(
    parser: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register vector database discovery commands."""
    vector_db = parser.add_parser(
        "vector-db", help="Discover and test configured vector databases"
    )
    commands = vector_db.add_subparsers(dest="vector_db_command", required=True)
    commands.add_parser("list", help="List vector database definitions")
    inspect = commands.add_parser("inspect", help="Inspect one vector database")
    inspect.add_argument("vector_db")
    test = commands.add_parser("test", help="Test one vector database configuration")
    test.add_argument("vector_db")


def _definitions() -> dict[str, tuple[str, ComponentDefinition]]:
    """Return configured vector backend definitions by stable component ID."""
    registry = load_configuration_registry()
    return {
        component_id: (category, definition)
        for component_id, (category, definition) in registry.components.items()
        if category == "vector_backend"
    }


def _resolve(requested: str) -> tuple[str, ComponentDefinition]:
    """Resolve a vector backend by component ID or implementation name."""
    definitions = _definitions()
    if requested in definitions:
        return requested, definitions[requested][1]
    matches = [
        (component_id, definition)
        for component_id, (_, definition) in definitions.items()
        if definition.implementation == requested
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"Unknown vector database: {requested}")
    raise ValueError(f"Vector database name is ambiguous: {requested}")


def _availability(implementation: str, config: dict[str, object]) -> dict[str, object]:
    """Describe configured resources without opening a provider connection."""
    if implementation == "qdrant":
        return {
            "configured": bool(os.getenv("QDRANT_URL", "").strip()),
            "url_configured": bool(os.getenv("QDRANT_URL", "").strip()),
            "collection": os.getenv("QDRANT_COLLECTION", "llm_grounding_v1"),
        }
    if implementation == "pgvector":
        return {
            "configured": bool(os.getenv("DATABASE_URL", "").strip()),
            "database_configured": bool(os.getenv("DATABASE_URL", "").strip()),
        }
    root = Path(FaissSettings.from_environment().root).expanduser()
    return {"configured": True, "storage_root": str(root), "exists": root.exists()}


def _payload(
    component_id: str, definition: ComponentDefinition
) -> dict[str, object]:
    """Build a safe vector backend discovery record."""
    factory = VectorStoreFactory()
    return {
        "id": component_id,
        "type": definition.type,
        "implementation": definition.implementation,
        "description": definition.description,
        "config": definition.config,
        "registered": definition.implementation in factory.implementations,
        "availability": _availability(definition.implementation, definition.config),
    }


async def _test_qdrant() -> tuple[str, str, str]:
    """Check Qdrant collection access without changing collection state."""
    from qdrant_client import AsyncQdrantClient

    settings = QdrantSettings.from_environment()
    client = AsyncQdrantClient(url=settings.url, api_key=settings.api_key)
    try:
        await client.get_collections()
    finally:
        await client.close()
    return "connected", "qdrant", settings.collection


async def _test_pgvector() -> tuple[str, str, str]:
    """Check PostgreSQL connectivity without creating vector tables."""
    from sqlalchemy import text

    from llm_grounding.infrastructure.database.session import create_session_factory

    engine, sessions = create_session_factory()
    try:
        async with sessions() as session:
            await session.execute(text("SELECT 1"))
    finally:
        await engine.dispose()
    return "connected", "pgvector", "postgresql"


async def execute(args: argparse.Namespace) -> int:
    """Execute vector backend discovery or a non-mutating connectivity test."""
    if args.vector_db_command == "list":
        rows: list[list[object]] = []
        for component_id, definition in sorted(
            ((key, value[1]) for key, value in _definitions().items())
        ):
            availability = _availability(definition.implementation, definition.config)
            rows.append(
                [
                    component_id,
                    definition.implementation,
                    availability["configured"],
                ]
            )
        print_table(["ID", "IMPLEMENTATION", "CONFIGURED"], rows)
        return 0
    component_id, definition = _resolve(args.vector_db)
    if args.vector_db_command == "inspect":
        print_json(_payload(component_id, definition))
        return 0
    if args.vector_db_command == "test":
        implementation = definition.implementation
        if implementation == "qdrant":
            result = await _test_qdrant()
        elif implementation == "pgvector":
            result = await _test_pgvector()
        elif implementation == "faiss":
            root = Path(FaissSettings.from_environment().root).expanduser()
            if root.exists() and root.is_dir():
                result = ("available", "faiss", str(root))
            else:
                result = ("ready", "faiss", str(root))
        else:
            raise ValueError(
                "Unsupported vector database implementation: "
                f"{implementation}"
            )
        print_table(["STATUS", "IMPLEMENTATION", "DETAIL"], [result])
        return 0
    raise ValueError(f"Unsupported vector database command: {args.vector_db_command}")
