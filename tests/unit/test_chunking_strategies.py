"""Verify deterministic token, hierarchy, structural, and sentence chunkers."""

from llm_grounding.domain.evidence import EvidenceType, EvidenceUnit
from llm_grounding.infrastructure.config.loader import load_configuration_registry
from llm_grounding.ingestion.models import ChunkerConfig
from llm_grounding.knowledge.chunking.hierarchical import HierarchicalChunker
from llm_grounding.knowledge.chunking.recursive_token import RecursiveTokenChunker
from llm_grounding.knowledge.chunking.sentence_window import SentenceWindowChunker
from llm_grounding.knowledge.chunking.structural import StructuralChunker
from llm_grounding.orchestration.registry import ComponentRegistry


def _evidence() -> tuple[EvidenceUnit, ...]:
    """Create ordered blocks with structural metadata and multilingual text."""
    return (
        EvidenceUnit(
            evidence_id="e1",
            document_id="document",
            evidence_type=EvidenceType.TEXT,
            source_uri="file:///document.pdf",
            order=0,
            page_number=1,
            block_index=0,
            text="Methods",
            metadata={"structure_type": "heading", "heading_level": 1},
        ),
        EvidenceUnit(
            evidence_id="e2",
            document_id="document",
            evidence_type=EvidenceType.TEXT,
            source_uri="file:///document.pdf",
            order=1,
            page_number=1,
            block_index=1,
            text="One sentence. Второе предложение! これは三番目です。",
        ),
        EvidenceUnit(
            evidence_id="e3",
            document_id="document",
            evidence_type=EvidenceType.TEXT,
            source_uri="file:///document.pdf",
            order=2,
            page_number=2,
            block_index=2,
            text="A | B\n--|--\n1 | 2",
            metadata={"structure_type": "table"},
        ),
    )


async def test_recursive_token_chunker_is_bounded_and_deterministic() -> None:
    """Recursive chunks stay within the configured limit and repeat identically."""
    config = ChunkerConfig(
        implementation="recursive_token",
        target_tokens=5,
        max_tokens=7,
        overlap_tokens=1,
    )
    chunker = RecursiveTokenChunker(config)

    first = await chunker.chunk(_evidence(), "ingestion", "document.pdf")
    second = await chunker.chunk(_evidence(), "ingestion", "document.pdf")

    assert first == second
    assert first
    assert all(item.metadata["token_count"] <= 7 for item in first)
    assert all(item.evidence_refs for item in first)


async def test_hierarchical_chunker_links_children_to_parent_chunks() -> None:
    """Every child carries a persisted parent ID and hierarchy path."""
    config = ChunkerConfig(
        implementation="hierarchical",
        hierarchical_levels=(8, 3),
        target_tokens=8,
        overlap_tokens=0,
    )
    chunks = await HierarchicalChunker(config).chunk(
        _evidence(), "ingestion", "document.pdf"
    )
    ids = {chunk.chunk_id for chunk in chunks}

    assert {chunk.metadata["chunk_level"] for chunk in chunks} == {0, 1}
    assert all(
        chunk.metadata["parent_chunk_id"] is None
        or chunk.metadata["parent_chunk_id"] in ids
        for chunk in chunks
    )
    assert all(chunk.metadata["hierarchy_path"] for chunk in chunks)


async def test_structural_chunker_preserves_structural_metadata() -> None:
    """Headings and tables remain visible as ordered structural chunks."""
    config = ChunkerConfig(
        implementation="structural", target_chars=64, overlap_chars=0
    )

    chunks = await StructuralChunker(config).chunk(
        _evidence(), "ingestion", "document.pdf"
    )

    assert chunks[0].metadata["structure_type"] == "heading"
    assert chunks[1].metadata["heading_hierarchy"] == ["Methods"]
    assert chunks[-1].metadata["structure_type"] == "table"
    assert chunks[-1].text == "A | B\n--|--\n1 | 2"


async def test_sentence_window_keeps_index_text_separate_from_context() -> None:
    """Sentence chunks index one sentence while metadata retains neighboring text."""
    config = ChunkerConfig(
        implementation="sentence_window",
        sentence_window=1,
        overlap_chars=0,
    )
    chunks = await SentenceWindowChunker(config).chunk(
        _evidence(), "ingestion", "document.pdf"
    )

    assert chunks
    assert chunks[1].text == "Второе предложение!"
    assert "One sentence." in chunks[1].metadata["context_text"]
    assert chunks[1].metadata["sentence_start_offset"] < chunks[1].metadata[
        "sentence_end_offset"
    ]
    assert chunks[1].evidence_refs[0].start > 0


def test_chunker_yaml_definitions_are_registered_and_buildable() -> None:
    """All shipped deterministic and model-backed definitions resolve."""
    configuration = load_configuration_registry()
    registry = ComponentRegistry()
    expected = {
        "chunker_recursive_tokens": "recursive_token",
        "chunker_hierarchical": "hierarchical",
        "chunker_structural": "structural",
        "chunker_sentence_window": "sentence_window",
        "chunker_semantic": "semantic",
        "chunker_llm_guided": "llm_guided",
        "chunker_late": "late_chunking",
    }

    for component_id, implementation in expected.items():
        definition = configuration.component("chunker", component_id)
        assert definition.implementation == implementation
        config = configuration._chunker(component_id)
        if implementation in {"semantic", "llm_guided", "late_chunking"}:
            assert config.model_id
        else:
            assert registry.build_chunker(config).__class__.__name__
