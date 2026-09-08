"""Offline tests for Phase 1 chunks, vector isolation, and grounded source validation."""

from pathlib import Path

import pytest
from pydantic import BaseModel

from llm_grounding.assessment.source_reference import SourceReferenceAssessor
from llm_grounding.decision.basic_grounding import BasicGroundingPolicy
from llm_grounding.domain.answers import DecisionAction
from llm_grounding.domain.corpus import (
    ChunkRetrievalResult,
    CorpusRequest,
    EvidenceReference,
    GroundingContext,
    RetrievalUnit,
)
from llm_grounding.domain.evidence import EvidenceType, EvidenceUnit
from llm_grounding.embedding.fake import DeterministicFakeEmbedder
from llm_grounding.embedding.models import EmbeddingVector
from llm_grounding.infrastructure.config.loader import (
    load_architecture_config,
    load_ingestion_config,
)
from llm_grounding.infrastructure.vector.memory import InMemoryVectorStore
from llm_grounding.ingestion.models import ChunkerConfig
from llm_grounding.knowledge.chunking.block_aware import BlockAwareChunker
from llm_grounding.knowledge.chunking.fixed_window import FixedWindowChunker
from llm_grounding.orchestration.basic_rag import BasicRagRunner
from llm_grounding.reasoning.basic_grounded import (
    BasicGroundedReasoner,
    GroundedResponsePayload,
)
from llm_grounding.rendering.grounded_answer import GroundedAnswerRenderer
from llm_grounding.retrieval.vector import VectorRetriever


class FixedGroundedModel:
    """Return preconfigured grounded output without any external model invocation."""

    def __init__(self, payload: GroundedResponsePayload) -> None:
        """Store the next schema-valid output and prompt history."""
        self.payload, self.prompts = payload, []

    async def generate[OutputT: BaseModel](
        self, prompt: str, output_type: type[OutputT]
    ) -> OutputT:
        """Record the prompt and validate output with the caller's requested schema."""
        self.prompts.append(prompt)
        return output_type.model_validate(self.payload.model_dump())


def evidence() -> tuple[EvidenceUnit, ...]:
    """Create adjacent canonical evidence with page and ordering provenance."""
    return (
        EvidenceUnit(
            "e1",
            "document-a",
            EvidenceType.TEXT,
            "file:///a.pdf",
            0,
            1,
            0,
            text="Alpha protocol uses photons.",
        ),
        EvidenceUnit(
            "e2",
            "document-a",
            EvidenceType.TEXT,
            "file:///a.pdf",
            1,
            2,
            1,
            text="Beta protocol uses entanglement.",
        ),
    )


async def test_chunkers_preserve_evidence_provenance_and_overlap() -> None:
    """Fixed/block-aware chunking retains source IDs and page spans without flattening identity."""
    config = ChunkerConfig(target_chars=32, overlap_chars=8)
    fixed = await FixedWindowChunker(config).chunk(evidence(), "ingestion", "paper.pdf")
    blocks = await BlockAwareChunker(config).chunk(evidence(), "ingestion", "paper.pdf")
    assert fixed[0].chunk_id == "ingestion:c000000"
    assert all(chunk.evidence_refs for chunk in fixed)
    assert all(
        reference.evidence_id in {"e1", "e2"}
        for chunk in fixed
        for reference in chunk.evidence_refs
    )
    assert all(
        chunk.page_start is not None and chunk.page_end is not None for chunk in blocks
    )
    assert fixed[0].text[-8:] == fixed[1].text[:8]


async def test_vector_store_enforces_project_isolation_and_embedding_compatibility() -> (
    None
):
    """Search predicates select only active project chunks before score ranking."""
    embedder = DeterministicFakeEmbedder()
    store = InMemoryVectorStore()
    alpha = RetrievalUnit(
        "a",
        "d1",
        "i1",
        "alpha.pdf",
        "Alpha photon protocol",
        0,
        1,
        1,
        (EvidenceReference("e1", 0, 21),),
    )
    beta = RetrievalUnit(
        "b",
        "d2",
        "i2",
        "beta.pdf",
        "Beta entanglement protocol",
        0,
        1,
        1,
        (EvidenceReference("e2", 0, 27),),
    )
    await store.upsert(
        "project-a", (alpha,), await embedder.embed_documents([alpha.text])
    )
    await store.upsert(
        "project-b", (beta,), await embedder.embed_documents([beta.text])
    )
    await store.associate_ingestion("project-c", "i1")
    results = await store.search(
        "project-a", await embedder.embed_query("entanglement protocol"), 5
    )
    assert [item.chunk.chunk_id for item in results] == ["a"]
    reused = await store.search(
        "project-c", await embedder.embed_query("photon protocol"), 5
    )
    assert [item.chunk.chunk_id for item in reused] == ["a"]
    with pytest.raises(ValueError, match="incompatible"):
        await store.search("project-a", EmbeddingVector((0.0,) * 16, "other"), 1)


async def test_unknown_model_source_ids_are_rejected_before_rendering() -> None:
    """A model cannot turn an arbitrary chunk ID into a displayed citation."""
    chunk = RetrievalUnit(
        "known",
        "document",
        "ingestion",
        "paper.pdf",
        "Supported statement.",
        0,
        1,
        1,
        (EvidenceReference("e", 0, 20),),
    )
    context = GroundingContext(
        CorpusRequest("project", "question"),
        (ChunkRetrievalResult(chunk, 1, "vector", 0.9),),
    )
    model = FixedGroundedModel(
        GroundedResponsePayload(
            answer_text="Answer",
            claims=[{"text": "Claim", "source_chunk_ids": ["invented"]}],
            insufficient_evidence=False,
        )
    )
    result = await BasicGroundedReasoner(model).reason(context)
    assessments = await SourceReferenceAssessor().assess_context(result, context)
    assert assessments[0].label == "fail"
    decision = await BasicGroundingPolicy().decide(result, assessments)
    assert decision.action == DecisionAction.REJECT
    with pytest.raises(ValueError, match="rejected"):
        await GroundedAnswerRenderer().render_context(decision, context)


async def test_basic_rag_renders_real_source_listing_and_never_history_as_evidence() -> (
    None
):
    """The query runner keeps history separate and renderer resolves known chunk provenance."""
    embedder = DeterministicFakeEmbedder()
    store = InMemoryVectorStore()
    chunk = RetrievalUnit(
        "known",
        "document",
        "ingestion",
        "paper.pdf",
        "The protocol uses photons.",
        0,
        4,
        4,
        (EvidenceReference("e", 0, 26),),
    )
    await store.upsert(
        "project", (chunk,), await embedder.embed_documents([chunk.text])
    )
    model = FixedGroundedModel(
        GroundedResponsePayload(
            answer_text="It uses photons.",
            claims=[
                {"text": "The protocol uses photons.", "source_chunk_ids": ["known"]}
            ],
            insufficient_evidence=False,
        )
    )
    runner = BasicRagRunner(
        VectorRetriever(embedder, store, 3),
        _UnusedSummaryRetriever(),
        BasicGroundedReasoner(model),
        SourceReferenceAssessor(),
        BasicGroundingPolicy(),
        GroundedAnswerRenderer(),
    )
    _, answer, _ = await runner.run(CorpusRequest("project", "What does it use?"))
    assert "[known]" in answer.text
    assert "Document: paper.pdf" in answer.text
    assert answer.evidence_ids == ("known",)
    assert "[SOURCE known]" in model.prompts[0]


class _UnusedSummaryRetriever:
    """Make non-summary runner tests fail if they take the summary branch."""

    async def retrieve(
        self, request: CorpusRequest
    ) -> tuple[ChunkRetrievalResult, ...]:
        """Raise because this test requests ordinary question answering."""
        raise AssertionError("Summary retrieval was not expected")


def test_configs_load_and_reingestion_identity_changes_with_chunking_or_embeddings() -> (
    None
):
    """Architecture/ingestion choices are typed, persisted, and identity-bearing."""
    architecture = load_architecture_config(
        Path("configs/architectures/basic_rag_v1.yaml")
    )
    ingestion = load_ingestion_config(Path("configs/ingestion/default.yaml"))
    assert architecture.name == "basic_rag_v1"
    assert architecture.retriever.type == "vector"
    assert (
        ingestion.identity()
        != ingestion.model_copy(
            update={"chunker": ChunkerConfig(target_chars=900, overlap_chars=120)}
        ).identity()
    )
    assert (
        ingestion.embedding.identity()
        != ingestion.embedding.model_copy(update={"model": "other"}).identity()
    )
