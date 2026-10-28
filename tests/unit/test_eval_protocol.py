"""Contract coverage for canonical evaluation observations and replay."""

from datetime import UTC, datetime
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from llm_grounding.api.eval import create_eval_router
from llm_grounding.application.chat_service import ChatService
from llm_grounding.application.eval_service import EvalService, EvalServiceError
from llm_grounding.domain.answers import Answer
from llm_grounding.domain.corpus import (
    Chat,
    ChunkRetrievalResult,
    CorpusRequest,
    EvidenceReference,
    Message,
    RetrievalUnit,
)
from llm_grounding.domain.evidence import RankingScore
from llm_grounding.observability.builder import ObservationBuilder
from llm_grounding.observability.models import RagUsage, RetrievalStageType
from llm_grounding.orchestration.retrieval_stages import (
    build_retrieval_stage_graph,
    build_retrieval_stages,
)
from llm_grounding.reasoning.basic_grounded import GroundedResponsePayload


def _chunk(chunk_id: str, page: int) -> ChunkRetrievalResult:
    """Create a scored canonical chunk with stable source provenance."""
    return ChunkRetrievalResult(
        RetrievalUnit(
            chunk_id,
            "document-1",
            "ingestion-1",
            "paper.pdf",
            f"Evidence text for {chunk_id}.",
            page,
            page,
            page,
            (EvidenceReference(f"evidence-{chunk_id}", 0, 12),),
        ),
        page,
        "vector",
        0.5 + page / 10,
        RankingScore(0.5 + page / 10, "cosine_similarity", "vector"),
    )


def test_observation_preserves_retrieval_rerank_and_final_context_contract() -> None:
    """The final-context stage contains exactly the evidence sent to generation."""
    first = _chunk("chunk-1", 1)
    second = _chunk("chunk-2", 2)
    reranked = (
        second.__class__(
            second.chunk,
            1,
            second.retriever_id,
            second.score,
            second.ranking_score,
            RankingScore(0.99, "cross_encoder_logit", "reranker"),
        ),
        first,
    )
    stages = build_retrieval_stages(
        (first, second),
        run_id="run-1",
        reranked_context=reranked,
        final_context=(reranked[0],),
    )
    stage_graph = build_retrieval_stage_graph(
        (first, second),
        run_id="run-1",
        reranked_context=reranked,
        final_context=(reranked[0],),
    )
    payload = GroundedResponsePayload(
        answer_text="Supported answer.",
        claims=[{"text": "Supported answer.", "source_chunk_ids": ["chunk-2"]}],
        insufficient_evidence=False,
    )
    observation = ObservationBuilder().from_basic_rag_metadata(
        run_id="run-1",
        request_id="request-1",
        corpus_id="project-1",
        architecture_name="query_basic_v1",
        architecture_version=1,
        request=CorpusRequest("project-1", "Question"),
        chunks=(reranked[0],),
        answer=Answer("Supported answer.", payload=payload.model_dump()),
        metadata={
            "status": "completed",
            "retrieval_stages": [item.model_dump(mode="json") for item in stages],
            "stage_graph": [item.model_dump(mode="json") for item in stage_graph],
            "usage": RagUsage(
                retrieval_calls=1,
                reranker_calls=1,
                usage_origin="unavailable",
            ).model_dump(mode="json"),
            "assessments": [
                {
                    "assessor": "relevance",
                    "dimension": "relevance",
                    "score": 0.8,
                    "label": "pass",
                    "evidence_ids": ["chunk-2"],
                }
            ],
            "decision": "accept",
            "decision_metadata": {"reason": "validated"},
        },
    )

    assert [item.stage_type for item in observation.retrieval] == [
        RetrievalStageType.CANDIDATE_RETRIEVAL,
        RetrievalStageType.RERANK,
        RetrievalStageType.FINAL_CONTEXT,
    ]
    assert [item.retrieval_id for item in observation.retrieval[-1].items] == [
        "chunk-2"
    ]
    assert observation.retrieval[1].items[0].rerank_score is not None
    assert observation.retrieval[1].items[0].rerank_score.type == "cross_encoder_logit"
    assert observation.usage.retrieval_calls == 1
    assert observation.usage.reranker_calls == 1
    assert observation.assessments[0].evidence_ids == ["chunk-2"]
    assert observation.decision is not None
    assert observation.decision.action == "accept"
    assert observation.answer is not None
    assert observation.answer.citations[0].source_evidence_id == "evidence-chunk-2"


class ObservationRepository:
    """Minimal repository double for observation replay and browser persistence."""

    def __init__(self) -> None:
        """Initialize empty observation and chat state."""
        self.observations = {}
        self.chat = Chat("chat-1", "project-1", "New chat", datetime.now(UTC))
        self.messages: list[Message] = []

    async def get_observation(self, run_id: str):
        """Return a previously persisted observation without running anything."""
        return self.observations.get(run_id)

    async def save_observation(self, observation) -> None:
        """Capture the canonical observation produced by a browser query."""
        self.observations[observation.run_id] = observation

    async def get_chat(self, chat_id: str) -> Chat:
        """Return the single test chat."""
        assert chat_id == self.chat.id
        return self.chat

    async def add_message(
        self,
        chat_id: str,
        role: str,
        content: str,
        rag_run_id: str | None = None,
        turn_title: str | None = None,
    ) -> Message:
        """Persist a test message."""
        message = Message(
            f"message-{len(self.messages)}",
            chat_id,
            role,
            content,
            datetime.now(UTC),
            rag_run_id,
            turn_title,
        )
        self.messages.append(message)
        return message

    async def recent_messages(self, chat_id: str, limit: int) -> tuple[Message, ...]:
        """Return recent test messages."""
        return tuple(self.messages[-limit:])

    async def chunks_by_ids(self, chunk_ids: tuple[str, ...]):
        """Return no chunks for the empty-retrieval browser contract test."""
        return {}

    async def record_run(self, *args: object, **kwargs: object) -> None:
        """Accept browser run metadata without a database."""

    async def set_generated_chat_name(self, chat_id: str, name: str) -> Chat:
        """Accept generated names for the browser service contract."""
        self.chat = Chat(
            self.chat.id,
            self.chat.project_id,
            name,
            self.chat.created_at,
            True,
        )
        return self.chat


class FixedBrowserRunner:
    """Return a deterministic browser-compatible answer without retrieval."""

    async def run(self, request, history, *, generate_chat_name=False):
        """Return an abstaining answer and complete execution metadata."""
        del request, history, generate_chat_name
        return (
            "browser-run",
            Answer(
                "Insufficient evidence.",
                payload={
                    "chat_name": "Browser query",
                    "turn_title": "Question",
                    "answer_text": "Insufficient evidence.",
                    "claims": [],
                    "insufficient_evidence": True,
                    "limitations": None,
                },
            ),
            {"status": "completed", "retrieved_chunk_ids": []},
        )


async def test_browser_query_persists_canonical_observation() -> None:
    """Browser execution stores an observation without changing its response DTO."""
    repository = ObservationRepository()
    service = ChatService(repository, FixedBrowserRunner())  # type: ignore[arg-type]

    answer = await service.ask("chat-1", "What is supported?")

    assert answer.metadata["rag_run_id"] == "browser-run"
    assert repository.observations["browser-run"].request_id == "browser-run"
    assert repository.observations["browser-run"].answer is not None


async def test_observation_lookup_does_not_execute_and_reports_missing_runs() -> None:
    """Run-ID replay reads persistence only and returns a stable 404 otherwise."""
    repository = ObservationRepository()
    service = EvalService(SimpleNamespace(repository=repository))  # type: ignore[arg-type]
    with pytest.raises(EvalServiceError, match="No observation") as error:
        await service.observation("missing")
    assert error.value.status_code == 404


async def test_observation_endpoint_returns_persisted_payload() -> None:
    """The protocol replay endpoint returns the original observation JSON."""
    repository = ObservationRepository()
    service = EvalService(SimpleNamespace(repository=repository))  # type: ignore[arg-type]
    observation = ObservationBuilder().build(
        run_id="run-1",
        request_id="request-1",
        corpus_id="corpus-1",
        architecture_name="query_basic_v1",
        architecture_version=1,
        request=CorpusRequest("corpus-1", "Question"),
    )
    repository.observations["run-1"] = observation
    app = FastAPI()
    app.include_router(create_eval_router(service))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/eval/v1/observations/run-1")

    assert response.status_code == 200
    assert response.json()["run_id"] == "run-1"
    assert response.json()["request_id"] == "request-1"
