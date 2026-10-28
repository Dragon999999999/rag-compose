"""Verify unified pipeline validation and representation compatibility identities."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from llm_grounding.application.pipeline_compatibility import (
    IndexCapabilities,
    validate_query_compatibility,
)
from llm_grounding.application.pipeline_sync import PipelineSynchronizer
from llm_grounding.domain.corpus import Pipeline
from llm_grounding.infrastructure.config.declarative import DeclarativePipelineConfig
from llm_grounding.infrastructure.config.loader import (
    ConfigurationError,
    load_model_catalog,
    load_pipeline_config,
)
from llm_grounding.infrastructure.config.models import ModelCatalog, PipelineConfig
from llm_grounding.infrastructure.config.registry import ConfigurationRegistry
from llm_grounding.infrastructure.database.models import PipelineRow
from llm_grounding.ingestion.models import (
    EmbeddingConfig,
    EmbeddingSpecification,
    IngestionConfig,
)
from llm_grounding.orchestration.registry import ComponentRegistry


def test_representation_hash_uses_effective_knowledge_builder_defaults() -> None:
    """Implicit and explicitly serialized defaults identify one representation."""
    catalog = load_model_catalog()
    fresh = load_pipeline_config(
        Path("config/pipelines/ingestion_default_v1.yaml"), catalog=catalog
    ).with_resolved_catalog(catalog)
    restored = PipelineConfig.model_validate(fresh.model_dump(mode="json"))

    assert fresh.representation_hash(catalog) == restored.representation_hash(catalog)
    assert fresh.config_hash == restored.config_hash


def test_representation_hash_survives_yaml_json_pydantic_round_trip() -> None:
    """Persisted JSON must retain the same effective ingestion identity as YAML."""
    catalog = load_model_catalog()
    fresh = load_pipeline_config(
        Path("config/pipelines/ingestion_default_v1.yaml"), catalog=catalog
    ).with_resolved_catalog(catalog)
    serialized = fresh.model_dump_json()
    restored = PipelineConfig.model_validate_json(serialized)

    assert restored.representation_hash(catalog) == fresh.representation_hash(catalog)
    assert restored.config_hash == fresh.config_hash


def test_representation_hash_changes_for_knowledge_builder_behavior() -> None:
    """A behavior-changing knowledge setting remains representation-affecting."""
    catalog = load_model_catalog()
    baseline = load_pipeline_config(
        Path("config/pipelines/ingestion_default_v1.yaml"), catalog=catalog
    ).with_resolved_catalog(catalog)
    assert baseline.representation is not None
    assert baseline.representation.knowledge_builder is not None
    changed_builder = baseline.representation.knowledge_builder.model_copy(
        update={"max_keywords": 13}
    )
    changed = baseline.model_copy(
        update={
            "representation": baseline.representation.model_copy(
                update={"knowledge_builder": changed_builder}
            )
        }
    )

    assert changed.representation_hash(catalog) != baseline.representation_hash(catalog)
    assert changed.config_hash != baseline.config_hash


def test_all_registered_ingestion_presets_have_round_trip_stable_hashes() -> None:
    """Every shipped ingestion preset is stable across persisted JSON restoration."""
    root = Path(__file__).resolve().parents[2]
    catalog = load_model_catalog()
    configuration_registry = ConfigurationRegistry.from_directories(
        root / "config/models", root / "config/components"
    )
    paths = sorted((root / "config/pipelines").rglob("*.yaml"))
    ingestion_configs: list[PipelineConfig] = []
    for path in paths:
        source = DeclarativePipelineConfig.model_validate(
            yaml.safe_load(path.read_text(encoding="utf-8"))
        )
        config = configuration_registry.pipeline(source, catalog)
        if config.kind == "ingestion":
            ingestion_configs.append(config.with_resolved_catalog(catalog))

    assert ingestion_configs
    for config in ingestion_configs:
        restored = PipelineConfig.model_validate(config.model_dump(mode="json"))
        assert restored.representation_hash(catalog) == config.representation_hash(
            catalog
        )
        assert restored.config_hash == config.config_hash


def test_all_checked_in_pipelines_validate_against_real_catalog() -> None:
    """Every shipped pipeline resolves only registered components and models."""
    catalog = load_model_catalog()
    paths = sorted(Path("config/pipelines").glob("*.yaml"))

    assert len(paths) == 3
    configs = [load_pipeline_config(path, catalog=catalog) for path in paths]
    assert {config.name for config in configs} == {
        "ingestion_default_v1",
        "ingestion_minimal_v1",
        "query_basic_v1",
    }


def test_aqueduct_embedding_component_resolves_into_ingestion_config(
    tmp_path: Path,
) -> None:
    """The Aqueduct embedding component can be selected by an ingestion pipeline."""
    source = Path("config/pipelines/ingestion_default_v1.yaml").read_text(
        encoding="utf-8"
    )
    path = tmp_path / "aqueduct-ingestion.yaml"
    path.write_text(
        source.replace(
            "embedding: embedding_local_minilm",
            "embedding: embedding_aqueduct_qwen",
        ),
        encoding="utf-8",
    )

    config = load_pipeline_config(path)
    embedding = config.resolve_ingestion(load_model_catalog()).embedding

    assert embedding.type == "openai_compatible"
    assert embedding.provider == "aqueduct"
    assert embedding.model == "qwen3-embedding-4b"
    assert embedding.dimension == 2560


@pytest.mark.parametrize(
    "knowledge_builder_yaml",
    ["", "  knowledge_builder: null\n"],
    ids=["omitted", "explicit-null"],
)
def test_ingestion_pipeline_can_omit_knowledge_builder(
    tmp_path: Path, knowledge_builder_yaml: str
) -> None:
    """Omitted and explicit null builders resolve to the parser normalization path."""
    source = Path("config/pipelines/ingestion_default_v1.yaml").read_text(
        encoding="utf-8"
    )
    path = tmp_path / "without-knowledge-builder.yaml"
    path.write_text(
        source.replace(
            "  knowledge_builder: knowledge_basic\n", knowledge_builder_yaml
        ),
        encoding="utf-8",
    )

    config = load_pipeline_config(path)
    resolved = config.resolve_ingestion(load_model_catalog())

    assert config.representation is not None
    assert config.representation.knowledge_builder is None
    assert resolved.knowledge_builder is None


@pytest.mark.parametrize(
    "dense_yaml",
    [
        "  embedding: null\n  vector_backend: null\n",
        "",
    ],
    ids=["explicit-null", "omitted"],
)
def test_ingestion_pipeline_can_disable_dense_index(
    tmp_path: Path, dense_yaml: str
) -> None:
    """PostgreSQL-only representation configurations resolve without dense stages."""
    source = Path("config/pipelines/ingestion_default_v1.yaml").read_text(
        encoding="utf-8"
    )
    path = tmp_path / "postgres-only-ingestion.yaml"
    path.write_text(
        source.replace(
            "  embedding: embedding_local_minilm\n  vector_backend: "
            "vector_backend_qdrant\n",
            dense_yaml,
        ),
        encoding="utf-8",
    )

    config = load_pipeline_config(path)
    resolved = config.resolve_ingestion(load_model_catalog())

    assert config.representation is not None
    assert config.representation.embedding is None
    assert config.representation.vector_backend is None
    assert resolved.embedding is None
    assert resolved.vector_backend is None


def test_embedding_without_backend_uses_application_default(tmp_path: Path) -> None:
    """An omitted backend is resolved through the configured defaults file."""
    source = Path("config/pipelines/ingestion_default_v1.yaml").read_text(
        encoding="utf-8"
    ).replace("  vector_backend: vector_backend_qdrant\n", "  vector_backend: null\n")
    path = tmp_path / "default-backend-ingestion.yaml"
    path.write_text(source, encoding="utf-8")
    defaults = tmp_path / "defaults.yaml"
    defaults.write_text(
        "schema_version: 1\ndefaults:\n  vector_backend: vector_backend_qdrant\n",
        encoding="utf-8",
    )

    config = load_pipeline_config(path, defaults_path=defaults)
    resolved = config.resolve_ingestion(load_model_catalog())

    assert config.representation is not None
    assert config.representation.vector_backend is not None
    assert config.representation.vector_backend.type == "qdrant"
    assert resolved.vector_backend == "qdrant"


def test_vector_backend_without_embedding_is_rejected(tmp_path: Path) -> None:
    """A vector backend cannot be configured for PostgreSQL-only ingestion."""
    source = Path("config/pipelines/ingestion_default_v1.yaml").read_text(
        encoding="utf-8"
    ).replace("  embedding: embedding_local_minilm\n", "  embedding: null\n")
    path = tmp_path / "invalid-vector-only-ingestion.yaml"
    path.write_text(source, encoding="utf-8")

    with pytest.raises(ConfigurationError, match="vector_backend requires"):
        load_pipeline_config(path)


def test_incompatible_embedding_representation_is_rejected() -> None:
    """The configured backend must support the selected embedding representation."""
    catalog = load_model_catalog().model_copy(
        update={
            "embedding_models": {
                "embedding_001": EmbeddingConfig(representation="sparse")
            }
        }
    )

    with pytest.raises(ConfigurationError, match="does not support sparse embeddings"):
        load_pipeline_config(
            Path("config/pipelines/ingestion_default_v1.yaml"), catalog=catalog
        )


def test_postgresql_only_capabilities_do_not_advertise_vectors() -> None:
    """Capability snapshots distinguish persisted chunks from dense indexes."""
    capabilities = IndexCapabilities.from_ingestion(
        IngestionConfig(embedding=None, vector_backend=None)
    )

    assert "chunk_vectors" not in capabilities.capabilities
    assert capabilities.embedding_identity is None
    assert capabilities.dimension is None


def test_retriever_requirements_are_loaded_from_component_yaml() -> None:
    """The executable dense retriever declares its resource requirements in YAML."""
    config = load_pipeline_config(Path("config/pipelines/query_basic_v1.yaml"))

    assert config.query is not None
    assert config.query.retriever.requirements.required == (
        "dense_vector_index",
        "compatible_query_embedder",
    )


def test_postgresql_only_project_cannot_select_dense_retriever() -> None:
    """Declared dense requirements are checked against actual project resources."""
    query = load_pipeline_config(Path("config/pipelines/query_basic_v1.yaml"))
    index = IndexCapabilities.from_ingestion(
        IngestionConfig(embedding=None, vector_backend=None)
    )

    result = validate_query_compatibility(query, index, ComponentRegistry())

    assert not result.compatible
    assert "dense_vector_index" in result.missing_capabilities


def test_unimplemented_retriever_is_not_advertised_as_executable() -> None:
    """Resource availability does not register a planned retrieval algorithm."""
    query = load_pipeline_config(Path("config/pipelines/query_basic_v1.yaml"))
    assert query.query is not None
    planned = query.model_copy(
        update={
            "query": query.query.model_copy(
                update={
                    "retriever": query.query.retriever.model_copy(
                        update={"implementation": "unimplemented"}
                    )
                }
            )
        }
    )

    with pytest.raises(ConfigurationError, match="Unknown retriever"):
        ComponentRegistry().validate_pipeline(planned, load_model_catalog())


def test_embedding_space_compatibility_ignores_provider_transport() -> None:
    """Compatible providers share a vector identity despite different transports."""
    local = EmbeddingConfig(
        type="fake",
        provider="local_gpu",
        model="bge-m3",
        embedding_space_id="bge-m3-dense-v1",
        dimension=1024,
    )
    remote = EmbeddingConfig(
        type="openai_compatible",
        provider="remote_api",
        model="bge-m3",
        embedding_space_id="bge-m3-dense-v1",
        dimension=1024,
        base_url="https://embeddings.example.test/v1",
    )

    assert local.identity() != remote.identity()
    assert local.vector_identity() == remote.vector_identity()
    assert local.is_compatible_with(remote)


def test_embedding_space_compatibility_rejects_equal_dimension_mismatch() -> None:
    """Dimensions alone do not establish compatibility between embedding spaces."""
    left = EmbeddingConfig(
        type="fake", model="model-a", embedding_space_id="space-a", dimension=384
    )
    right = EmbeddingConfig(
        type="fake", model="model-b", embedding_space_id="space-b", dimension=384
    )

    assert not left.is_compatible_with(right)


def test_query_embedder_compatibility_uses_the_indexed_embedding_spec() -> None:
    """An explicitly selected query provider must match persisted space metadata."""
    ingestion = EmbeddingConfig(
        type="fake",
        provider="local",
        model="shared",
        embedding_space_id="shared-v1",
        dimension=3,
    )
    query_embedding = ingestion.model_copy(update={"provider": "remote"})
    catalog = ModelCatalog(embedding_models={"query_provider": query_embedding})
    base_query = load_pipeline_config(Path("config/pipelines/query_basic_v1.yaml"))
    assert base_query.query is not None
    query = base_query.model_copy(
        update={
            "query": base_query.query.model_copy(
                update={
                    "retriever": base_query.query.retriever.model_copy(
                        update={"query_embedder": "query_provider"}
                    )
                }
            )
        }
    )
    index = IndexCapabilities.from_ingestion(
        IngestionConfig(embedding=ingestion, vector_backend="qdrant")
    )

    result = validate_query_compatibility(
        query, index, ComponentRegistry(), catalog
    )

    assert result.compatible
    assert index.embedding_specification == EmbeddingSpecification.model_validate(
        index.as_dict()["embedding_specification"]
    )


def test_representation_changes_produce_distinct_index_identity() -> None:
    """Chunking changes cannot silently reuse an existing representation index."""
    catalog = load_model_catalog()
    base = load_pipeline_config(
        Path("config/pipelines/ingestion_default_v1.yaml"), catalog=catalog
    )
    changed = base.model_copy(
        update={
            "representation": base.representation.model_copy(
                update={
                    "chunker": base.representation.chunker.model_copy(
                        update={"target_chars": 600}
                    )
                }
            )
        }
    )

    assert base.representation_hash(catalog) != changed.representation_hash(catalog)



def test_ingestion_and_query_pipeline_specs_resolve_separately() -> None:
    """Specialized definitions compose through a project's effective ingestion."""
    catalog = load_model_catalog()
    ingestion = load_pipeline_config(
        Path("config/pipelines/ingestion_default_v1.yaml"), catalog=catalog
    )
    query = load_pipeline_config(
        Path("config/pipelines/query_basic_v1.yaml"), catalog=catalog
    )

    assert ingestion.kind == "ingestion"
    assert ingestion.query is None
    assert query.kind == "query"
    assert query.representation is None
    effective_ingestion = ingestion.resolve_ingestion(catalog)
    architecture, resolved_ingestion = query.resolve_query(catalog, effective_ingestion)
    assert architecture.retriever.type == "dense"
    assert resolved_ingestion.identity() == effective_ingestion.identity()


def test_query_compatibility_uses_index_capabilities_not_dimension_alone() -> None:
    """A query pipeline requires the actual vector capability and embedding identity."""
    catalog = load_model_catalog()
    query = load_pipeline_config(
        Path("config/pipelines/query_basic_v1.yaml"), catalog=catalog
    )
    ingestion = load_pipeline_config(
        Path("config/pipelines/ingestion_default_v1.yaml"), catalog=catalog
    ).resolve_ingestion(catalog)
    compatible = validate_query_compatibility(
        query,
        IndexCapabilities.from_ingestion(ingestion),
        ComponentRegistry(),
    )
    dimension_only = IndexCapabilities(frozenset({"chunk_vectors"}), None, 384)
    incompatible = validate_query_compatibility(
        query, dimension_only, ComponentRegistry()
    )

    assert compatible.compatible
    assert not incompatible.compatible
    assert "embedding_space" in incompatible.missing_capabilities


def test_pipeline_loader_rejects_unknown_components_without_echoing_secret(
    tmp_path: Path,
) -> None:
    """Component references remain closed and invalid values are not leaked."""
    source = Path("config/pipelines/ingestion_default_v1.yaml").read_text(
        encoding="utf-8"
    )
    path = tmp_path / "invalid.yaml"
    path.write_text(
        source.replace("parser: parser_pymupdf", "parser: secret-model"),
        encoding="utf-8",
    )

    with pytest.raises(
        ConfigurationError, match="Invalid pipeline component or model reference"
    ) as caught:
        load_pipeline_config(path)

    assert "secret-model" not in str(caught.value)


def test_pipeline_row_persists_jsonb_configuration_and_ownership_fields() -> None:
    """The ORM contract includes source ownership and representation identity."""
    columns = PipelineRow.__table__.c

    assert columns.config_json.type.__class__.__name__ == "JSONB"
    assert {"config_hash", "representation_hash", "source_kind", "kind"}.issubset(
        columns.keys()
    )
    assert {"active", "is_default", "updated_at"}.issubset(columns.keys())


def test_registry_keeps_pipeline_registration_closed() -> None:
    """A pipeline cannot select arbitrary Python import paths."""
    config = load_pipeline_config(Path("config/pipelines/query_basic_v1.yaml"))
    settings = config.model_dump()
    settings["query"]["renderer"]["implementation"] = "python:os.system"

    with pytest.raises(ConfigurationError, match="Unknown renderer component"):
        ComponentRegistry().validate_pipeline(
            type(config).model_validate(settings), load_model_catalog()
        )


class RecordingPipelineRepository:
    """Record synchronizer calls without requiring a PostgreSQL service."""

    def __init__(self) -> None:
        """Start with an empty persisted pipeline collection."""
        self.records: dict[str, Pipeline] = {}
        self.versions: dict[tuple[str, int], str] = {}
        self.upsert_count = 0

    async def list_pipelines(
        self, *, active_only: bool = False, kind: str | None = None
    ) -> tuple[Pipeline, ...]:
        """Return persisted records for the synchronizer preflight check."""
        records = tuple(self.records.values())
        if active_only:
            records = tuple(record for record in records if record.active)
        if kind is not None:
            records = tuple(record for record in records if record.kind == kind)
        return records

    async def get_pipeline_by_name(self, name: str) -> Pipeline | None:
        """Return the current record for a stable pipeline name."""
        return self.records.get(name)

    async def upsert_pipeline_from_yaml(
        self,
        config,
        *,
        source_path: str,
        source_fingerprint: str,
        representation_hash: str,
    ) -> Pipeline:
        """Apply the synchronizer's YAML-owned upsert contract."""
        self.upsert_count += 1
        previous = self.records.get(config.name)
        if previous is not None and previous.source_kind == "api":
            return previous
        pipeline_id = previous.id if previous is not None else config.name
        version_key = (pipeline_id, config.version)
        stored_hash = self.versions.get(version_key)
        if stored_hash is not None and stored_hash != config.config_hash:
            raise ValueError(
                f"Pipeline {pipeline_id} version {config.version} already contains "
                "a different configuration; increment the pipeline version"
            )
        now = datetime.now(UTC)
        record = Pipeline(
            pipeline_id,
            config.name,
            config.version,
            config.schema_version,
            config.description,
            config.model_dump(mode="json"),
            config.config_hash,
            representation_hash,
            "yaml",
            source_path,
            source_fingerprint,
            True,
            config.is_default,
            previous.created_at if previous is not None else now,
            now,
        )
        self.records[config.name] = record
        self.versions[version_key] = config.config_hash
        return record


async def test_startup_synchronizer_discovers_and_fingerprints_yaml(
    tmp_path: Path,
) -> None:
    """Startup imports new files and remains idempotent on the next pass."""
    source = Path("config/pipelines/query_basic_v1.yaml")
    (tmp_path / source.name).write_text(
        source.read_text(encoding="utf-8"), encoding="utf-8"
    )
    repository = RecordingPipelineRepository()
    synchronizer = PipelineSynchronizer(repository, tmp_path, load_model_catalog())

    first = await synchronizer.synchronize()
    second = await synchronizer.synchronize()

    assert first.imported == 1
    assert first.updated == 0
    assert first.pipelines[0].source_fingerprint
    assert second.imported == 0
    assert second.updated == 0
    assert repository.upsert_count == 1


async def test_startup_sync_rejects_changed_yaml_without_version_bump(
    tmp_path: Path,
) -> None:
    """A changed YAML document cannot overwrite an immutable version snapshot."""
    source = Path("config/pipelines/query_basic_v1.yaml")
    pipeline_path = tmp_path / source.name
    pipeline_path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    repository = RecordingPipelineRepository()
    synchronizer = PipelineSynchronizer(repository, tmp_path, load_model_catalog())

    await synchronizer.synchronize()
    pipeline_path.write_text(
        pipeline_path.read_text(encoding="utf-8").replace(
            "description: Default query pipeline for indexed project documents.",
            "description: Changed query pipeline for indexed project documents.",
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"Pipeline query_basic_v1 version 1 already contains a different",
    ):
        await synchronizer.synchronize()

    assert repository.records["query_basic_v1"].version == 1
    assert (
        repository.records["query_basic_v1"].config["query"]["retriever"]["top_k"]
        == 8
    )


async def test_startup_sync_accepts_changed_yaml_after_version_bump(
    tmp_path: Path,
) -> None:
    """A changed YAML document imports normally once its version is incremented."""
    source = Path("config/pipelines/query_basic_v1.yaml")
    pipeline_path = tmp_path / source.name
    pipeline_path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    repository = RecordingPipelineRepository()
    synchronizer = PipelineSynchronizer(repository, tmp_path, load_model_catalog())

    await synchronizer.synchronize()
    changed = pipeline_path.read_text(encoding="utf-8")
    pipeline_path.write_text(
        changed.replace("\nversion: 1\n", "\nversion: 2\n", 1).replace(
            "description: Default query pipeline for indexed project documents.",
            "description: Changed query pipeline for indexed project documents.",
        ),
        encoding="utf-8",
    )

    updated = await synchronizer.synchronize()
    repeated = await synchronizer.synchronize()

    assert updated.updated == 1
    assert repeated.updated == 0
    assert repository.records["query_basic_v1"].version == 2
    assert (
        repository.records["query_basic_v1"].config["query"]["retriever"]["top_k"]
        == 8
    )
