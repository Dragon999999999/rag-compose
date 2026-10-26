"""Test Phase 1 chunks, vector isolation, and source validation offline."""

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel, ValidationError

from llm_grounding.assessment.source_reference import SourceReferenceAssessor
from llm_grounding.decision.basic_grounding import BasicGroundingPolicy
from llm_grounding.domain.answers import DecisionAction, DecisionResult, ReasoningResult
from llm_grounding.domain.corpus import (
    ChunkRetrievalResult,
    CorpusRequest,
    EvidenceReference,
    GroundingContext,
    Message,
    RetrievalUnit,
)
from llm_grounding.domain.evidence import EvidenceType, EvidenceUnit
from llm_grounding.embedding.fake import DeterministicFakeEmbedder
from llm_grounding.embedding.models import EmbeddingVector
from llm_grounding.infrastructure.config.loader import (
    load_ingestion_config,
    load_model_catalog,
    load_pipeline_config,
)
from llm_grounding.infrastructure.prompts import PromptLibrary
from llm_grounding.infrastructure.vector.memory import InMemoryVectorStore
from llm_grounding.ingestion.models import ChunkerConfig
from llm_grounding.knowledge.chunking.block_aware import BlockAwareChunker
from llm_grounding.knowledge.chunking.fixed_window import FixedWindowChunker
from llm_grounding.orchestration.basic_rag import BasicRagRunner
from llm_grounding.reasoning.basic_grounded import (
    BasicGroundedReasoner,
    GroundedResponsePayload,
    ModelGroundedResponsePayload,
    build_grounded_prompt_context,
)
from llm_grounding.rendering.grounded_answer import GroundedAnswerRenderer
from llm_grounding.retrieval.vector import VectorRetriever


class FixedGroundedModel:
    """Return preconfigured grounded output without any external model invocation."""

    def __init__(self, payload: ModelGroundedResponsePayload) -> None:
        """Store the next schema-valid output and prompt history."""
        self.payload, self.prompts = payload, []

    async def generate[OutputT: BaseModel](
        self, prompt: str, output_type: type[OutputT]
    ) -> OutputT:
        """Record the prompt and validate output with the caller's requested schema."""
        self.prompts.append(prompt)
        return output_type.model_validate(self.payload.model_dump())


def grounded_reasoner(model: FixedGroundedModel) -> BasicGroundedReasoner:
    """Create a Basic RAG reasoner using the configured grounded QA prompt."""
    return BasicGroundedReasoner(model, PromptLibrary.default(), "grounded_qa_v1")


class RecordingFakeEmbedder:
    """Record which embedding operation a query pipeline requests."""

    def __init__(self) -> None:
        """Wrap the deterministic test embedder and initialize call records."""
        self._delegate = DeterministicFakeEmbedder()
        self.document_inputs: list[list[str]] = []
        self.query_inputs: list[str] = []

    @property
    def model_identity(self) -> str:
        """Expose the wrapped embedding-space identity."""
        return self._delegate.model_identity

    async def embed_documents(self, texts: list[str]) -> tuple[EmbeddingVector, ...]:
        """Record and embed document chunks."""
        self.document_inputs.append(texts)
        return await self._delegate.embed_documents(texts)

    async def embed_query(self, text: str) -> EmbeddingVector:
        """Record and embed one query."""
        self.query_inputs.append(text)
        return await self._delegate.embed_query(text)


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
    """Chunking retains source IDs and page spans without flattening identity."""
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


async def test_vector_store_enforces_isolation_and_embedding_compatibility() -> None:
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


async def test_unknown_model_source_refs_are_rejected_before_rendering() -> None:
    """A model cannot turn an unknown local reference into a citation."""
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
        ModelGroundedResponsePayload(
            turn_title="Unsupported source reference",
            answer_text="Answer",
            claims=[{"text": "Claim", "source_refs": ["chunk_99"]}],
            insufficient_evidence=False,
        )
    )
    result = await grounded_reasoner(model).reason(context)
    assessments = await SourceReferenceAssessor().assess_context(result, context)
    assert assessments[0].label == "fail"
    decision = await BasicGroundingPolicy().decide(result, assessments)
    assert decision.action == DecisionAction.REJECT
    with pytest.raises(ValueError, match="rejected"):
        await GroundedAnswerRenderer().render_context(decision, context)


async def test_local_source_ref_is_canonicalized_before_assessment() -> None:
    """A request-local alias resolves to the retrieved canonical ID."""
    canonical_id = "709e24e2-fa17-44e8-bfab-3cf256e584f0:c000016"
    chunk = RetrievalUnit(
        canonical_id,
        "document",
        "ingestion",
        "paper.pdf",
        "The framework is the paper's main contribution.",
        0,
        1,
        1,
        (EvidenceReference("e", 0, 47),),
    )
    context = GroundingContext(
        CorpusRequest("project", "What is the main contribution?"),
        (ChunkRetrievalResult(chunk, 1, "vector", 0.9),),
        generate_chat_name=True,
    )
    model = FixedGroundedModel(
        ModelGroundedResponsePayload(
            chat_name="Grounded Paper Contribution",
            turn_title="Paper contribution",
            answer_text="The paper introduces a framework.",
            claims=[
                {
                    "text": "The paper introduces a framework.",
                    "source_refs": ["chunk_1"],
                }
            ],
            insufficient_evidence=False,
        )
    )

    reasoning = await grounded_reasoner(model).reason(context)
    payload = GroundedResponsePayload.model_validate(reasoning.payload)
    assessments = await SourceReferenceAssessor().assess_context(reasoning, context)

    assert payload.turn_title == "Paper contribution"
    assert payload.chat_name == "Grounded Paper Contribution"
    assert payload.claims[0].source_chunk_ids == [canonical_id]
    assert reasoning.metadata["raw_model_payload"] == {
        "chat_name": "Grounded Paper Contribution",
        "turn_title": "Paper contribution",
        "answer_text": "The paper introduces a framework.",
        "claims": [
            {
                "text": "The paper introduces a framework.",
                "source_refs": ["chunk_1"],
            }
        ],
        "insufficient_evidence": False,
        "limitations": None,
    }
    assert reasoning.metadata["source_ref_aliases"] == {"chunk_1": canonical_id}
    assert reasoning.metadata["source_ref_resolutions"] == [
        {"source_ref": "chunk_1", "canonical_chunk_id": canonical_id}
    ]
    assert assessments[0].label == "pass"
    assert assessments[0].evidence_ids == (canonical_id,)
    assert canonical_id not in model.prompts[0]
    assert '<chunk ref="chunk_1">' in model.prompts[0]
    assert reasoning.metadata["prompt_id"] == "grounded_qa_v1"
    assert reasoning.metadata["prompt_version"] == "1.3"


async def test_first_chat_answer_requires_generated_chat_name() -> None:
    """An unset chat cannot silently remain unnamed after model generation."""
    chunk = RetrievalUnit(
        "canonical-a", "doc-a", "ing-a", "a.pdf", "Evidence", 0, 1, 1, ()
    )
    context = GroundingContext(
        CorpusRequest("project", "question"),
        (ChunkRetrievalResult(chunk, 1, "vector", 0.9),),
        generate_chat_name=True,
    )
    model = FixedGroundedModel(
        ModelGroundedResponsePayload(
            turn_title="Question",
            answer_text="Answer",
            claims=[{"text": "Answer", "source_refs": ["chunk_1"]}],
            insufficient_evidence=False,
        )
    )

    with pytest.raises(ValidationError, match="chat_name"):
        await grounded_reasoner(model).reason(context)


def test_model_claim_requires_at_least_one_source_ref() -> None:
    """Supported factual claims cannot use an empty model-facing reference list."""
    with pytest.raises(ValidationError, match="at least 1 item"):
        ModelGroundedResponsePayload(
            turn_title="Uncited claim",
            answer_text="Answer",
            claims=[{"text": "Uncited claim", "source_refs": []}],
            insufficient_evidence=False,
        )


def test_model_response_requires_a_navigation_title() -> None:
    """The model must return a title during the existing grounded-answer call."""
    with pytest.raises(ValidationError, match="turn_title"):
        ModelGroundedResponsePayload(
            answer_text="Answer",
            claims=[{"text": "Cited claim", "source_refs": ["chunk_1"]}],
            insufficient_evidence=False,
        )


def test_model_chat_name_rejects_default_placeholder() -> None:
    """The model cannot finalize an unnamed chat with another generic placeholder."""
    with pytest.raises(ValidationError, match="default placeholder"):
        ModelGroundedResponsePayload(
            chat_name="New Chat",
            turn_title="Question",
            answer_text="Answer",
            claims=[{"text": "Cited claim", "source_refs": ["chunk_1"]}],
            insufficient_evidence=False,
        )


def test_model_requires_claims_when_evidence_is_sufficient() -> None:
    """A non-abstaining response cannot bypass the claim-level source contract."""
    with pytest.raises(ValidationError, match="claims must not be empty"):
        ModelGroundedResponsePayload(
            turn_title="Missing claims",
            answer_text="Answer",
            claims=[],
            insufficient_evidence=False,
        )


def test_prompt_aliases_follow_final_chunk_order() -> None:
    """Aliases are deterministic and preserve post-retrieval ordering."""
    first = RetrievalUnit("canonical-a", "doc-a", "ing-a", "a.pdf", "A", 0, 1, 1, ())
    second = RetrievalUnit("canonical-b", "doc-b", "ing-b", "b.pdf", "B", 1, 2, 2, ())
    context = GroundingContext(
        CorpusRequest("project", "question"),
        (
            ChunkRetrievalResult(first, 1, "vector", 0.9),
            ChunkRetrievalResult(second, 2, "vector", 0.8),
        ),
    )

    prompt_context = build_grounded_prompt_context(context)

    assert [chunk.ref for chunk in prompt_context.chunks] == ["chunk_1", "chunk_2"]
    assert prompt_context.alias_to_chunk_id == {
        "chunk_1": "canonical-a",
        "chunk_2": "canonical-b",
    }


async def test_multiple_source_refs_resolve_to_canonical_ids() -> None:
    """Every valid alias is resolved before canonical source assessment."""
    first = RetrievalUnit("canonical-a", "doc-a", "ing-a", "a.pdf", "A", 0, 1, 1, ())
    second = RetrievalUnit("canonical-b", "doc-b", "ing-b", "b.pdf", "B", 1, 2, 2, ())
    context = GroundingContext(
        CorpusRequest("project", "question"),
        (
            ChunkRetrievalResult(first, 1, "vector", 0.9),
            ChunkRetrievalResult(second, 2, "vector", 0.8),
        ),
    )
    model = FixedGroundedModel(
        ModelGroundedResponsePayload(
            turn_title="Combined evidence",
            answer_text="A and B.",
            claims=[{"text": "A and B.", "source_refs": ["chunk_1", "chunk_2"]}],
            insufficient_evidence=False,
        )
    )

    reasoning = await grounded_reasoner(model).reason(context)
    payload = GroundedResponsePayload.model_validate(reasoning.payload)
    assessments = await SourceReferenceAssessor().assess_context(reasoning, context)

    assert payload.claims[0].source_chunk_ids == ["canonical-a", "canonical-b"]
    assert assessments[0].label == "pass"
    assert assessments[0].evidence_ids == ("canonical-a", "canonical-b")


async def test_history_is_context_not_a_source_reference() -> None:
    """Conversation text is rendered for intent but never aliases as evidence."""
    chunk = RetrievalUnit(
        "canonical-a", "doc-a", "ing-a", "a.pdf", "Evidence text.", 0, 1, 1, ()
    )
    context = GroundingContext(
        CorpusRequest("project", "question"),
        (ChunkRetrievalResult(chunk, 1, "vector", 0.9),),
        (
            Message(
                "message",
                "chat",
                "user",
                "Earlier conversation context.",
                datetime.now(UTC),
            ),
        ),
    )
    model = FixedGroundedModel(
        ModelGroundedResponsePayload(
            turn_title="Evidence answer",
            answer_text="Evidence answer.",
            claims=[{"text": "Evidence answer.", "source_refs": ["chunk_1"]}],
            insufficient_evidence=False,
        )
    )

    reasoning = await grounded_reasoner(model).reason(context)

    assert "Earlier conversation context." in model.prompts[0]
    assert reasoning.metadata["source_ref_aliases"] == {"chunk_1": "canonical-a"}


async def test_basic_rag_renders_sources_and_excludes_history_as_evidence() -> None:
    """The runner separates history while rendering known chunk provenance."""
    embedder = RecordingFakeEmbedder()
    store = InMemoryVectorStore()
    canonical_id = "ingestion:c000007"
    chunk = RetrievalUnit(
        canonical_id,
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
        ModelGroundedResponsePayload(
            turn_title="Protocol mechanism",
            answer_text="It uses photons.",
            claims=[
                {
                    "text": "The protocol uses photons.",
                    "source_refs": ["chunk_1"],
                }
            ],
            insufficient_evidence=False,
        )
    )
    runner = BasicRagRunner(
        VectorRetriever(embedder, store, 3),
        _UnusedSummaryRetriever(),
        grounded_reasoner(model),
        SourceReferenceAssessor(),
        BasicGroundingPolicy(),
        GroundedAnswerRenderer(),
    )
    _, answer, _ = await runner.run(CorpusRequest("project", "What does it use?"))
    assert "[paper_1]" in answer.text
    assert canonical_id not in answer.text
    assert "[USED] paper_1" in answer.text
    assert "Retrieved chunks" in answer.text
    assert "Document: paper.pdf" in answer.text
    assert answer.evidence_ids == (canonical_id,)
    assert '<chunk ref="chunk_1">' in model.prompts[0]
    assert canonical_id not in model.prompts[0]
    assert embedder.document_inputs == [["The protocol uses photons."]]
    assert embedder.query_inputs == ["What does it use?"]


async def test_renderer_lists_every_retrieved_chunk_with_stable_display_labels() -> (
    None
):
    """Cited and merely retrieved chunks remain distinguishable without UUIDs."""
    cited = RetrievalUnit(
        "canonical-cited",
        "document",
        "ingestion",
        "Smith.pdf",
        "Supported text.",
        14,
        3,
        3,
        (),
    )
    additional = RetrievalUnit(
        "canonical-additional",
        "document",
        "ingestion",
        "Smith.pdf",
        "Additional context.",
        15,
        4,
        4,
        (),
    )
    context = GroundingContext(
        CorpusRequest("project", "question"),
        (
            ChunkRetrievalResult(cited, 1, "vector", 0.9),
            ChunkRetrievalResult(additional, 2, "vector", 0.8),
        ),
    )
    payload = GroundedResponsePayload(
        turn_title="Question",
        answer_text="Supported text.",
        claims=[{"text": "Supported text.", "source_chunk_ids": [cited.chunk_id]}],
        insufficient_evidence=False,
    )

    decision = DecisionResult(
        DecisionAction.ACCEPT,
        ReasoningResult("result", payload.model_dump()),
    )
    answer = await GroundedAnswerRenderer().render_context(decision, context)

    assert "[Smith_15]" in answer.text
    assert "[USED] Smith_15" in answer.text
    assert "[retrieved] Smith_16" in answer.text
    assert "canonical-cited" not in answer.text
    assert "canonical-additional" not in answer.text


async def test_basic_rag_renders_explicit_insufficient_evidence() -> None:
    """A validated abstention renders as a normal insufficient-evidence answer."""
    embedder = DeterministicFakeEmbedder()
    store = InMemoryVectorStore()
    chunk = RetrievalUnit(
        "ingestion:c000001",
        "document",
        "ingestion",
        "paper.pdf",
        "The source discusses a different topic.",
        0,
        1,
        1,
        (EvidenceReference("e", 0, 38),),
    )
    await store.upsert(
        "project", (chunk,), await embedder.embed_documents([chunk.text])
    )
    model = FixedGroundedModel(
        ModelGroundedResponsePayload(
            turn_title="Evidence limitation",
            answer_text="The retrieved source does not answer the question.",
            claims=[],
            insufficient_evidence=True,
            limitations="No supporting passage was retrieved.",
        )
    )
    runner = BasicRagRunner(
        VectorRetriever(embedder, store, 3),
        _UnusedSummaryRetriever(),
        grounded_reasoner(model),
        SourceReferenceAssessor(),
        BasicGroundingPolicy(),
        GroundedAnswerRenderer(),
    )

    _, answer, metadata = await runner.run(
        CorpusRequest("project", "What is not covered?")
    )

    assert answer.text.startswith("Insufficient evidence:")
    assert "[retrieved] paper_1" in answer.text
    assert answer.metadata["insufficient_evidence"] is True
    assert metadata["decision"] == DecisionAction.ABSTAIN
    assert metadata["status"] == "completed"


async def test_rejected_decision_bypasses_grounded_renderer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unsafe model output produces a controlled failure outside the renderer."""
    embedder = DeterministicFakeEmbedder()
    store = InMemoryVectorStore()
    chunk = RetrievalUnit(
        "ingestion:c000001",
        "document",
        "ingestion",
        "paper.pdf",
        "Supported statement.",
        0,
        1,
        1,
        (EvidenceReference("e", 0, 20),),
    )
    await store.upsert(
        "project", (chunk,), await embedder.embed_documents([chunk.text])
    )
    model = FixedGroundedModel(
        ModelGroundedResponsePayload(
            chat_name="Question About Safety",
            turn_title="Unsafe answer",
            answer_text="Unsafe answer.",
            claims=[{"text": "Unsafe claim.", "source_refs": ["chunk_99"]}],
            insufficient_evidence=False,
        )
    )
    renderer = GroundedAnswerRenderer()
    render_context = AsyncMock(side_effect=AssertionError("renderer was called"))
    monkeypatch.setattr(renderer, "render_context", render_context)
    runner = BasicRagRunner(
        VectorRetriever(embedder, store, 3),
        _UnusedSummaryRetriever(),
        grounded_reasoner(model),
        SourceReferenceAssessor(),
        BasicGroundingPolicy(),
        renderer,
    )

    _, answer, metadata = await runner.run(
        CorpusRequest("project", "Question"), generate_chat_name=True
    )

    render_context.assert_not_awaited()
    assert answer.text.startswith("Grounding failure:")
    assert answer.metadata["grounding_rejected"] is True
    assert metadata["generated_chat_name"] == "Question About Safety"
    assert metadata["decision"] == DecisionAction.REJECT
    assert metadata["status"] == "rejected"


class _UnusedSummaryRetriever:
    """Make non-summary runner tests fail if they take the summary branch."""

    async def retrieve(
        self, request: CorpusRequest
    ) -> tuple[ChunkRetrievalResult, ...]:
        """Raise because this test requests ordinary question answering."""
        raise AssertionError("Summary retrieval was not expected")


def test_configs_and_reingestion_identity_track_representation_changes() -> None:
    """Architecture/ingestion choices are typed, persisted, and identity-bearing."""
    query = load_pipeline_config(Path("config/pipelines/query_basic_v1.yaml"))
    ingestion = load_ingestion_config(
        Path("config/pipelines/ingestion_default_v1.yaml")
    )
    architecture = query.resolve_query(load_model_catalog(), ingestion)[0]
    assert architecture.name == "query_basic_v1"
    assert architecture.retriever.type == "vector"
    assert architecture.reasoner.prompt_id == "grounded_qa_v1"
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
