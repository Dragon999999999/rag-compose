"""Basic RAG v1 query orchestration, separate from document ingestion and Baseline 0."""

from datetime import UTC, datetime
from uuid import uuid4

from llm_grounding.assessment.source_reference import SourceReferenceAssessor
from llm_grounding.decision.basic_grounding import BasicGroundingPolicy
from llm_grounding.domain.answers import Answer, DecisionAction, DecisionResult
from llm_grounding.domain.corpus import (
    ChunkRetrievalResult,
    CorpusRequest,
    GroundingContext,
    Message,
    TaskType,
)
from llm_grounding.observability.embedding import embedding_diagnostics
from llm_grounding.observability.models import LLMUsage
from llm_grounding.observability.trace import TraceRecorder
from llm_grounding.observability.usage import UsageAccumulator
from llm_grounding.orchestration.grounding_diagnostics import GroundingDiagnostics
from llm_grounding.orchestration.retrieval_stages import build_retrieval_stages
from llm_grounding.reasoning.basic_grounded import BasicGroundedReasoner
from llm_grounding.rendering.grounded_answer import GroundedAnswerRenderer
from llm_grounding.retrieval.vector import SummaryRetriever, VectorRetriever


class BasicRagRunner:
    """Run project-scoped retrieval, reasoning, validation, and rendering."""

    def __init__(
        self,
        vector: VectorRetriever,
        summary: SummaryRetriever,
        reasoner: BasicGroundedReasoner,
        assessor: SourceReferenceAssessor,
        policy: BasicGroundingPolicy,
        renderer: GroundedAnswerRenderer,
        diagnostics: GroundingDiagnostics | None = None,
    ) -> None:
        """Inject independently testable query stages."""
        self._vector, self._summary, self._reasoner = vector, summary, reasoner
        self._assessor, self._policy, self._renderer = assessor, policy, renderer
        self._diagnostics = diagnostics or GroundingDiagnostics()

    async def run(
        self,
        request: CorpusRequest,
        history: tuple[Message, ...] = (),
        *,
        generate_chat_name: bool = False,
    ) -> tuple[str, Answer, dict[str, object]]:
        """Execute Basic RAG and retain metadata suitable for persistence."""
        traces = TraceRecorder()
        usage = UsageAccumulator()
        total_span = traces.start("rag_e2e")
        retrieval_span = traces.start("retrieval")
        chunks = await self.retrieve(request)
        query_embedding = self._vector.last_query_embedding
        if query_embedding is not None:
            usage.add_embedding()
        traces.end(retrieval_span, output_count=len(chunks))
        if not chunks:
            traces.end(total_span, output_count=0)
            self._diagnostics.report_empty_retrieval(
                GroundingContext(request, (), history, generate_chat_name)
            )
            answer = Answer(
                "Insufficient evidence: no indexed project chunks matched "
                "this request.",
                metadata={"insufficient_evidence": True},
            )
            return (
                str(uuid4()),
                answer,
                {
                    "status": "completed",
                    "retrieved_chunk_ids": [],
                    "retrieval_results": [],
                    "retrieval_stages": [
                        stage.model_dump(mode="json")
                        for stage in build_retrieval_stages((), run_id=str(uuid4()))
                    ],
                    "trace": [span.model_dump(mode="json") for span in traces.spans],
                    "usage": usage.build().model_dump(mode="json"),
                    "query_embedding_id": (
                        query_embedding.embedding_id
                        if query_embedding is not None
                        else None
                    ),
                    "embedding": (
                        embedding_diagnostics(query_embedding).model_dump(mode="json")
                        if query_embedding is not None
                        else None
                    ),
                    "answer_payload": {},
                    "evidence_ids": [],
                    "created_at": datetime.now(UTC).isoformat(),
                },
            )
        context = GroundingContext(request, chunks, history, generate_chat_name)
        context_span = traces.start("final_context", input_count=len(chunks))
        traces.end(context_span, output_count=len(context.chunks))
        reasoning_span = traces.start("generation", input_count=len(context.chunks))
        reasoning = await self._reasoner.reason(context)
        generation_data = reasoning.metadata.get("generation")
        if isinstance(generation_data, dict):
            generation_usage = generation_data.get("usage")
            if isinstance(generation_usage, dict):
                usage.add_llm(LLMUsage.model_validate(generation_usage))
        traces.end(reasoning_span, output_count=1)
        assessment_span = traces.start("assessment")
        assessments = await self._assessor.assess_context(reasoning, context)
        traces.end(assessment_span, output_count=len(assessments))
        decision_span = traces.start("decision", input_count=len(assessments))
        decision = await self._policy.decide(reasoning, assessments)
        traces.end(decision_span, output_count=1)
        self._diagnostics.report(context, reasoning, assessments, decision)
        rejected = decision.action == DecisionAction.REJECT
        render_span = traces.start("render")
        answer = (
            self._rejected_answer(decision)
            if rejected
            else await self._renderer.render_context(decision, context)
        )
        traces.end(render_span, output_count=1)
        traces.end(total_span, output_count=1)
        return (
            reasoning.result_id,
            answer,
            {
                "status": (
                    "rejected" if rejected else "completed"
                ),
                "observation_status": (
                    "rejected"
                    if rejected
                    else "abstained"
                    if decision.action == DecisionAction.ABSTAIN
                    else "completed"
                ),
                "retrieved_chunk_ids": [item.chunk.chunk_id for item in chunks],
                "retrieval_results": [
                    {
                        "chunk_id": item.chunk.chunk_id,
                        "rank": item.rank,
                        "retriever_id": item.retriever_id,
                        "score": item.score,
                    }
                    for item in chunks
                ],
                "assessment_labels": [item.label for item in assessments],
                "decision": decision.action,
                "generated_chat_name": reasoning.payload.get("chat_name"),
                "answer_payload": answer.payload,
                "evidence_ids": list(answer.evidence_ids),
                "answer_metadata": answer.metadata,
                "retrieval_stages": [
                    stage.model_dump(mode="json")
                    for stage in build_retrieval_stages(
                        chunks, run_id=reasoning.result_id
                    )
                ],
                "trace": [span.model_dump(mode="json") for span in traces.spans],
                "usage": usage.build().model_dump(mode="json"),
                "query_embedding_id": (
                    query_embedding.embedding_id
                    if query_embedding is not None
                    else None
                ),
                "embedding": (
                    embedding_diagnostics(query_embedding).model_dump(mode="json")
                    if query_embedding is not None
                    else None
                ),
                "created_at": datetime.now(UTC).isoformat(),
            },
        )

    async def retrieve(
        self, request: CorpusRequest
    ) -> tuple[ChunkRetrievalResult, ...]:
        """Retrieve the exact candidate set used by the query path."""
        return await (
            self._summary.retrieve(request)
            if request.task_type == TaskType.SUMMARY
            else self._vector.retrieve(request)
        )

    @staticmethod
    def _rejected_answer(decision: DecisionResult) -> Answer:
        """Return a safe response without rendering rejected model content."""
        reason = str(decision.metadata.get("reason", "grounding_validation_failed"))
        return Answer(
            "Grounding failure: the generated response did not pass source validation.",
            metadata={"grounding_rejected": True, "reason": reason},
        )
