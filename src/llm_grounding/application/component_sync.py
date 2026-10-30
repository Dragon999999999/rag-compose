"""Synchronize validated standalone component definitions into PostgreSQL."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import yaml

from llm_grounding.infrastructure.config.loader import load_configuration_registry
from llm_grounding.infrastructure.database.repositories import PostgresRepository


@dataclass(frozen=True)
class ComponentSyncResult:
    """Summarize one component-definition import pass."""

    imported: int
    synchronized: int
    component_ids: tuple[str, ...]


class ComponentSynchronizer:
    """Validate component YAML and persist its effective source definitions."""

    def __init__(
        self,
        repository: PostgresRepository,
        directory: Path | None = None,
    ) -> None:
        """Configure the PostgreSQL repository and optional component directory."""
        self._repository = repository
        self._directory = directory

    async def synchronize(self) -> ComponentSyncResult:
        """Load schemas and definitions, then upsert every validated component."""
        registry = load_configuration_registry(components_directory=self._directory)
        paths = self._paths(self._directory)
        imported = 0
        for component_id, (category, definition) in registry.components.items():
            path = paths.get(component_id)
            if path is None:
                raise ValueError(f"No source path found for component {component_id}")
            await self._repository.upsert_component_definition(
                category,
                definition,
                source_path=str(path),
                source_fingerprint=sha256(
                    path.read_bytes()
                ).hexdigest(),
            )
            imported += 1
        return ComponentSyncResult(
            imported, imported, tuple(sorted(registry.components))
        )

    @staticmethod
    def _paths(directory: Path | None) -> dict[str, Path]:
        """Discover component source files without treating schemas as instances."""
        root = directory or Path(__file__).resolve().parents[3] / "config/components"
        paths: dict[str, Path] = {}
        for path in (*root.rglob("*.yaml"), *root.rglob("*.yml")):
            if path.name in {"schema.yaml", "schema.yml"}:
                continue
            value = yaml.safe_load(path.read_text(encoding="utf-8"))
            if isinstance(value, dict) and isinstance(value.get("id"), str):
                paths[value["id"]] = path
        return paths
