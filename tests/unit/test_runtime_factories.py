"""Verify component construction is routed through registered factories."""

from pathlib import Path

from llm_grounding.embedding.factory import EmbeddingProviderFactory
from llm_grounding.embedding.fake import DeterministicFakeEmbedder
from llm_grounding.infrastructure.config.loader import (
    load_model_catalog,
    load_pipeline_config,
)
from llm_grounding.infrastructure.config.models import (
    ComponentConfig,
    RetrieverConfig,
    VectorBackendConfig,
)
from llm_grounding.infrastructure.vector.factory import VectorStoreFactory
from llm_grounding.infrastructure.vector.memory import InMemoryVectorStore
from llm_grounding.ingestion.models import ChunkerConfig
from llm_grounding.knowledge.chunking.block_aware import BlockAwareChunker
from llm_grounding.orchestration.registry import (
    ComponentRegistry,
    RetrieverDependencies,
)
from llm_grounding.retrieval.full_context import FullContextRetriever
from llm_grounding.retrieval.hybrid import HybridRetriever
from llm_grounding.retrieval.sparse import SparseRetriever
from llm_grounding.retrieval.vector import VectorRetriever


def test_minimal_ingestion_pipeline_resolves_without_dense_dependencies() -> None:
    """The built-in minimal pipeline selects only parser, chunker, and PostgreSQL."""
    catalog = load_model_catalog()
    config = load_pipeline_config(
        Path("config/pipelines/ingestion_minimal_v1.yaml"), catalog=catalog
    )

    resolved = config.resolve_ingestion(catalog)

    assert config.is_default is False
    assert resolved.knowledge_builder is None
    assert resolved.embedding is None
    assert resolved.vector_backend is None


def test_component_registry_builds_all_retrievers_from_configuration() -> None:
    """Each retrieval strategy is constructed by the shared component registry."""
    registry = ComponentRegistry()
    dependencies = RetrieverDependencies(repository=object())  # type: ignore[arg-type]

    full_context = registry.build_retriever(
        RetrieverConfig(type="full_context"), dependencies
    )
    sparse = registry.build_retriever(
        RetrieverConfig(type="postgres_fts"), dependencies
    )
    assert isinstance(full_context, FullContextRetriever)
    assert isinstance(sparse, SparseRetriever)

    dense_dependencies = RetrieverDependencies(
        repository=object(),  # type: ignore[arg-type]
        vector_store=InMemoryVectorStore(),
        embedder=DeterministicFakeEmbedder(3, model_identity="space"),
    )
    dense = registry.build_retriever(
        RetrieverConfig(type="vector"), dense_dependencies
    )
    hybrid = registry.build_retriever(
        RetrieverConfig(type="hybrid_rrf"), dense_dependencies
    )
    assert isinstance(dense, VectorRetriever)
    assert isinstance(hybrid, HybridRetriever)


def test_parser_chunker_and_provider_factories_accept_registered_extensions() -> None:
    """New adapters can be registered without changing pipeline orchestration."""
    registry = ComponentRegistry()
    parser = registry.build_parser(ComponentConfig(type="pymupdf"))
    assert hasattr(parser, "parse")
    chunker = registry.build_chunker(
        ChunkerConfig(type="block_aware")
    )
    assert isinstance(chunker, BlockAwareChunker)

    embedding_factory = EmbeddingProviderFactory()
    embedding_factory.register(
        "test_provider",
        lambda config: DeterministicFakeEmbedder(
            config.dimension, model_identity=config.vector_identity()
        ),
    )
    embedding = embedding_factory.create(
        load_model_catalog().embedding("embedding_001")
    )
    assert embedding.model_identity

    vector_factory = VectorStoreFactory()
    vector_factory.register("test_backend", lambda repository: InMemoryVectorStore())
    vector_store = vector_factory.create(
        VectorBackendConfig(type="test_backend"), object()  # type: ignore[arg-type]
    )
    assert isinstance(vector_store, InMemoryVectorStore)
