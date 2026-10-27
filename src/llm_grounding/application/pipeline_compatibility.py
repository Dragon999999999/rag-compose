"""Capability-based compatibility checks between project indexes and query pipelines.

This module is the application boundary between persisted ingestion artifacts and
chat-level query composition. Query pipelines never provide an embedder; they can
only be selected when the project's actual indexed representation supplies the
capabilities required by their registered retriever.
"""

from dataclasses import dataclass

from llm_grounding.infrastructure.config.models import PipelineConfig
from llm_grounding.ingestion.models import IngestionConfig
from llm_grounding.orchestration.registry import ComponentRegistry


@dataclass(frozen=True)
class IndexCapabilities:
    """Describe the representation and backend features available to a project."""

    capabilities: frozenset[str]
    embedding_identity: str | None
    dimension: int | None
    ingestion_pipeline_id: str | None = None
    ingestion_pipeline_version: int | None = None
    ingestion_pipeline_config_hash: str | None = None
    representation_hash: str | None = None

    @classmethod
    def from_ingestion(
        cls,
        config: IngestionConfig,
        *,
        ingestion_pipeline_id: str | None = None,
        ingestion_pipeline_version: int | None = None,
        ingestion_pipeline_config_hash: str | None = None,
        representation_hash: str | None = None,
    ) -> "IndexCapabilities":
        """Infer capabilities from effective ingestion settings and backend support."""
        capabilities = {"full_text", "metadata_filtering"}
        if config.embedding is not None and config.vector_backend == "qdrant":
            capabilities.add("chunk_vectors")
        return cls(
            frozenset(capabilities),
            config.embedding.identity() if config.embedding is not None else None,
            config.embedding.dimension if config.embedding is not None else None,
            ingestion_pipeline_id,
            ingestion_pipeline_version,
            ingestion_pipeline_config_hash,
            representation_hash,
        )

    @classmethod
    def from_mapping(cls, value: dict[str, object]) -> "IndexCapabilities":
        """Deserialize capabilities retained on a project row."""
        raw_capabilities = value.get("capabilities", ())
        capabilities = frozenset(
            item for item in raw_capabilities if isinstance(item, str)
        ) if isinstance(raw_capabilities, (list, tuple, set)) else frozenset()
        dimension = value.get("dimension")
        return cls(
            capabilities,
            value.get("embedding_identity")
            if isinstance(value.get("embedding_identity"), str)
            else None,
            int(dimension) if isinstance(dimension, int) else None,
            value.get("ingestion_pipeline_id")
            if isinstance(value.get("ingestion_pipeline_id"), str)
            else None,
            int(value["ingestion_pipeline_version"])
            if isinstance(value.get("ingestion_pipeline_version"), int)
            else None,
            value.get("ingestion_pipeline_config_hash")
            if isinstance(value.get("ingestion_pipeline_config_hash"), str)
            else None,
            value.get("representation_hash")
            if isinstance(value.get("representation_hash"), str)
            else None,
        )

    def as_dict(self) -> dict[str, object]:
        """Serialize the non-secret index identity for persistence and APIs."""
        return {
            "capabilities": sorted(self.capabilities),
            "embedding_identity": self.embedding_identity,
            "dimension": self.dimension,
            "ingestion_pipeline_id": self.ingestion_pipeline_id,
            "ingestion_pipeline_version": self.ingestion_pipeline_version,
            "ingestion_pipeline_config_hash": self.ingestion_pipeline_config_hash,
            "representation_hash": self.representation_hash,
        }


@dataclass(frozen=True)
class CompatibilityResult:
    """Explain whether one query pipeline can execute against one project index."""

    compatible: bool
    required_capabilities: frozenset[str]
    missing_capabilities: frozenset[str]
    reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        """Serialize compatibility details as a stable API response."""
        return {
            "compatible": self.compatible,
            "required_capabilities": sorted(self.required_capabilities),
            "missing_capabilities": sorted(self.missing_capabilities),
            "reason": self.reason,
        }


def required_query_capabilities(
    config: PipelineConfig, registry: ComponentRegistry
) -> frozenset[str]:
    """Infer query requirements from registered retriever behavior."""
    if config.query is None:
        raise ValueError("A query pipeline requires query stages")
    return registry.query_capabilities(config.query.retriever.type)


def validate_query_compatibility(
    config: PipelineConfig,
    index: IndexCapabilities,
    registry: ComponentRegistry,
) -> CompatibilityResult:
    """Validate query requirements against an actual project's indexed capabilities."""
    required = required_query_capabilities(config, registry)
    missing = required - index.capabilities
    if missing:
        return CompatibilityResult(
            False,
            required,
            frozenset(missing),
            "The project index is missing required capabilities: "
            + ", ".join(sorted(missing)),
        )
    if "chunk_vectors" in required and (
        index.embedding_identity is None or index.dimension is None
    ):
        return CompatibilityResult(
            False,
            required,
            frozenset({"embedding_space"}),
            "The project does not have a complete indexed embedding-space identity.",
        )
    return CompatibilityResult(True, required, frozenset(), None)
