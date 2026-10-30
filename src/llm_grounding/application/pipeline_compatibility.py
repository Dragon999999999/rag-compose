"""Capability-based compatibility checks between project indexes and query pipelines.

This module is the application boundary between persisted ingestion artifacts and
chat-level query composition. Query pipelines never provide an embedder; they can
only be selected when the project's actual indexed representation supplies the
capabilities required by their registered retriever.
"""

from dataclasses import dataclass

from llm_grounding.infrastructure.config.models import ModelCatalog, PipelineConfig
from llm_grounding.ingestion.models import EmbeddingSpecification, IngestionConfig
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
    embedding_specification: EmbeddingSpecification | None = None

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
        capabilities = {
            "full_text",
            "metadata_filtering",
            "persisted_chunks",
            "postgres_fts",
        }
        if config.embedding is not None and config.vector_backend in {
            "qdrant",
            "pgvector",
            "faiss",
        }:
            representation = config.embedding.representation
            if representation == "dense":
                capabilities.update({"chunk_vectors", "dense_vector_index"})
            elif representation == "sparse":
                capabilities.update({"sparse_vector_index", "sparse_query_embedder"})
            elif representation == "multivector":
                capabilities.update(
                    {"multivector_index", "late_interaction_model"}
                )
        return cls(
            frozenset(capabilities),
            (
                config.embedding.vector_identity()
                if config.embedding is not None
                else None
            ),
            config.embedding.dimension if config.embedding is not None else None,
            ingestion_pipeline_id,
            ingestion_pipeline_version,
            ingestion_pipeline_config_hash,
            representation_hash,
            (
                config.embedding.specification()
                if config.embedding is not None
                and config.embedding.embedding_space_id is not None
                else None
            ),
        )

    @classmethod
    def from_mapping(cls, value: dict[str, object]) -> "IndexCapabilities":
        """Deserialize capabilities retained on a project row."""
        raw_capabilities = value.get("capabilities", ())
        capabilities = frozenset(
            item for item in raw_capabilities if isinstance(item, str)
        ) if isinstance(raw_capabilities, (list, tuple, set)) else frozenset()
        dimension = value.get("dimension")
        embedding_identity = value.get("embedding_identity")
        ingestion_pipeline_id = value.get("ingestion_pipeline_id")
        ingestion_pipeline_version = value.get("ingestion_pipeline_version")
        ingestion_pipeline_config_hash = value.get("ingestion_pipeline_config_hash")
        representation_hash = value.get("representation_hash")
        embedding_specification = value.get("embedding_specification")
        return cls(
            capabilities,
            embedding_identity if isinstance(embedding_identity, str) else None,
            int(dimension) if isinstance(dimension, int) else None,
            ingestion_pipeline_id if isinstance(ingestion_pipeline_id, str) else None,
            (
                int(ingestion_pipeline_version)
                if isinstance(ingestion_pipeline_version, int)
                else None
            ),
            (
                ingestion_pipeline_config_hash
                if isinstance(ingestion_pipeline_config_hash, str)
                else None
            ),
            representation_hash if isinstance(representation_hash, str) else None,
            (
                EmbeddingSpecification.model_validate(embedding_specification)
                if isinstance(embedding_specification, dict)
                else None
            ),
        )

    def as_dict(self) -> dict[str, object]:
        """Serialize the non-secret index identity for persistence and APIs."""
        return {
            "capabilities": sorted(self.capabilities),
            "embedding_identity": self.embedding_identity,
            "dimension": self.dimension,
            "embedding_specification": (
                self.embedding_specification.model_dump(mode="json")
                if self.embedding_specification is not None
                else None
            ),
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
    return registry.query_requirements(config.query.retriever)


def validate_query_compatibility(
    config: PipelineConfig,
    index: IndexCapabilities,
    registry: ComponentRegistry,
    catalog: ModelCatalog | None = None,
) -> CompatibilityResult:
    """Validate query requirements against an actual project's indexed capabilities."""
    required = required_query_capabilities(config, registry)
    resource_requirements = required - {"compatible_query_embedder"}
    available = set(index.capabilities)
    # ``full_text`` was the pre-retriever name for the persisted chunk store.
    # Treat it as the compatibility alias for snapshots created before Phase 4.
    if "full_text" in available:
        available.add("persisted_chunks")
    missing = resource_requirements - available
    if "compatible_query_embedder" in required and (
        index.embedding_identity is None or index.dimension is None
    ):
        missing = missing | {"embedding_space"}
    if missing:
        reason = (
            "The project does not have a complete indexed embedding-space identity."
            if "embedding_space" in missing
            else "The project index is missing required capabilities: "
            + ", ".join(sorted(missing))
        )
        return CompatibilityResult(
            False,
            required,
            frozenset(missing),
            reason,
        )
    if config.query is None:
        raise ValueError("A query pipeline requires query stages")
    requested = config.query.retriever.query_embedder
    if requested is not None:
        if catalog is None:
            return CompatibilityResult(
                False,
                required,
                frozenset({"compatible_query_embedder"}),
                "The query embedding provider could not be resolved from the "
                "model catalog.",
            )
        try:
            query_embedding = catalog.embedding(requested)
        except ValueError as exc:
            return CompatibilityResult(
                False,
                required,
                frozenset({"compatible_query_embedder"}),
                str(exc),
            )
        if index.embedding_specification is None:
            if query_embedding.vector_identity() != index.embedding_identity:
                return CompatibilityResult(
                    False,
                    required,
                    frozenset({"embedding_space"}),
                    "The explicit query embedder is incompatible with the legacy "
                    "indexed embedding identity.",
                )
        elif not query_embedding.is_compatible_with_spec(index.embedding_specification):
            return CompatibilityResult(
                False,
                required,
                frozenset({"embedding_space"}),
                "The explicit query embedder is incompatible with the indexed "
                "embedding space.",
            )
    return CompatibilityResult(True, required, frozenset(), None)
