"""Basic RAG v1 query orchestration, separate from document ingestion and Baseline 0."""

from datetime import UTC, datetime
from uuid import uuid4

from llm_grounding.assessment.source_reference import SourceReferenceAssessor
from llm_grounding.decision.basic_grounding import BasicGroundingPolicy
from llm_grounding.domain.answers import Answer
from llm_grounding.domain.corpus import (
    CorpusRequest,
    GroundingContext,
    Message,
    TaskType,
)
from llm_grounding.reasoning.basic_grounded import BasicGroundedReasoner
from llm_grounding.rendering.grounded_answer import GroundedAnswerRenderer
from llm_grounding.retrieval.vector import SummaryRetriever, VectorRetriever


class BasicRagRunner:
    """Retrieve project-scoped chunks, reason, validate IDs, decide, and render an Answer."""

    def __init__(
        self,
        vector: VectorRetriever,
        summary: SummaryRetriever,
        reasoner: BasicGroundedReasoner,
        assessor: SourceReferenceAssessor,
        policy: BasicGroundingPolicy,
        renderer: GroundedAnswerRenderer,
    ) -> None:
        """Inject independently testable query stages without invoking ingestion work."""
        self._vector, self._summary, self._reasoner = vector, summary, reasoner
        self._assessor, self._policy, self._renderer = assessor, policy, renderer

    async def run(
        self, request: CorpusRequest, history: tuple[Message, ...] = ()
    ) -> tuple[str, Answer, dict[str, object]]:
        """Execute Basic RAG v1 and retain source IDs/metadata suitable for run persistence."""
        chunks = await (
            self._summary.retrieve(request)
            if request.task_type == TaskType.SUMMARY
            else self._vector.retrieve(request)
        )
        if not chunks:
            answer = Answer(
                "Insufficient evidence: no indexed project chunks matched this request.",
                metadata={"insufficient_evidence": True},
            )
            return (
                str(uuid4()),
                answer,
                {
                    "status": "completed",
                    "retrieved_chunk_ids": [],
                    "created_at": datetime.now(UTC).isoformat(),
                },
            )
        context = GroundingContext(request, chunks, history)
        reasoning = await self._reasoner.reason(context)
        assessments = await self._assessor.assess_context(reasoning, context)
        decision = await self._policy.decide(reasoning, assessments)
        answer = await self._renderer.render_context(decision, context)
        return (
            reasoning.result_id,
            answer,
            {
                "status": "completed",
                "retrieved_chunk_ids": [item.chunk.chunk_id for item in chunks],
                "assessment_labels": [item.label for item in assessments],
                "decision": decision.action,
                "created_at": datetime.now(UTC).isoformat(),
            },
        )
