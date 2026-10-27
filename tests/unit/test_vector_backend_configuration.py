"""Configuration and factory coverage for selectable vector backends."""

from pathlib import Path

from llm_grounding.infrastructure.config.loader import (
    load_model_catalog,
    load_pipeline_config,
)
from llm_grounding.infrastructure.config.models import VectorBackendConfig
from llm_grounding.infrastructure.vector.factory import VectorStoreFactory
from llm_grounding.infrastructure.vector.faiss import FaissVectorStore
from llm_grounding.infrastructure.vector.pgvector import PgVectorStore


def test_registered_vector_backend_components_resolve_from_pipeline_yaml(
    tmp_path: Path,
) -> None:
    """The YAML registry accepts either new backend without code-level selection."""
    source = Path("config/pipelines/ingestion_default_v1.yaml").read_text(
        encoding="utf-8"
    )
    for backend_id, implementation in (
        ("vector_backend_pgvector", "pgvector"),
        ("vector_backend_faiss", "faiss"),
    ):
        path = tmp_path / f"{implementation}.yaml"
        path.write_text(
            source.replace("vector_backend_qdrant", backend_id), encoding="utf-8"
        )
        config = load_pipeline_config(path)
        assert config.representation is not None
        assert config.representation.vector_backend is not None
        assert config.representation.vector_backend.implementation == implementation
        assert config.resolve_ingestion(load_model_catalog()).vector_backend == (
            implementation
        )


def test_factory_constructs_new_backends_without_initializing_resources() -> None:
    """Factory construction is side-effect free for pgvector and FAISS."""
    factory = VectorStoreFactory()
    repository = object()  # type: ignore[assignment]
    pgvector = factory.create(
        VectorBackendConfig(implementation="pgvector"), repository  # type: ignore[arg-type]
    )
    faiss = factory.create(
        VectorBackendConfig(implementation="faiss"), repository  # type: ignore[arg-type]
    )
    assert isinstance(pgvector, PgVectorStore)
    assert isinstance(faiss, FaissVectorStore)
    assert faiss.loaded_index_count == 0
