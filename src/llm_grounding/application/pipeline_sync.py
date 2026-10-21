"""Import validated pipeline YAML into PostgreSQL at application startup.

YAML is treated as an explicitly owned import source. Once a pipeline has been
created or edited through an API, synchronization leaves that stable name alone.
"""

from dataclasses import dataclass
from pathlib import Path

from llm_grounding.domain.corpus import Pipeline
from llm_grounding.infrastructure.config.loader import (
    discover_pipeline_paths,
    load_model_catalog,
    load_pipeline_config,
    pipeline_source_fingerprint,
)
from llm_grounding.infrastructure.config.models import ModelCatalog
from llm_grounding.infrastructure.database.repositories import PostgresRepository


@dataclass(frozen=True)
class PipelineSyncResult:
    """Summarize one deterministic pipeline import pass."""

    pipelines: tuple[Pipeline, ...]
    imported: int
    updated: int
    skipped_api_owned: tuple[str, ...]


class PipelineSynchronizer:
    """Discover, validate, and persist pipeline definitions."""

    def __init__(
        self,
        repository: PostgresRepository,
        directory: Path | None = None,
        catalog: ModelCatalog | None = None,
    ) -> None:
        """Configure synchronization without opening providers or network clients."""
        self._repository = repository
        self._directory = directory
        self._catalog = catalog

    async def synchronize(self) -> PipelineSyncResult:
        """Import all pipeline files and return the current persisted records."""
        catalog = self._catalog or load_model_catalog()
        paths = discover_pipeline_paths(self._directory)
        imported = 0
        updated = 0
        skipped: list[str] = []
        records: list[Pipeline] = []
        for path in paths:
            config = load_pipeline_config(path, catalog=catalog)
            previous = await self._repository.get_pipeline_by_name(config.name)
            record = await self._repository.upsert_pipeline_from_yaml(
                config,
                source_path=str(path),
                source_fingerprint=pipeline_source_fingerprint(path),
                representation_hash=config.representation_hash(catalog),
            )
            records.append(record)
            if previous is None:
                imported += 1
            elif previous.source_kind == "api":
                skipped.append(config.name)
            elif (
                previous.config_hash != record.config_hash
                or previous.source_fingerprint != record.source_fingerprint
                or previous.representation_hash != record.representation_hash
            ):
                updated += 1
        return PipelineSyncResult(tuple(records), imported, updated, tuple(skipped))

    async def sync(self) -> PipelineSyncResult:
        """Alias for callers that prefer a concise lifecycle method."""
        return await self.synchronize()


async def synchronize_pipelines(
    repository: PostgresRepository,
    directory: Path | None = None,
    catalog: ModelCatalog | None = None,
) -> PipelineSyncResult:
    """Synchronize pipeline YAML through a repository-owned import boundary."""
    return await PipelineSynchronizer(repository, directory, catalog).synchronize()
