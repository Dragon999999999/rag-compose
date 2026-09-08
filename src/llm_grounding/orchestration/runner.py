"""Linear Baseline 0 orchestration, retaining success and failure run artifacts.

The execution strategy is local to this runner. Component protocols can be called
repeatedly by later strategies and do not prescribe a single retrieve/reason cycle.
"""

from collections.abc import Awaitable
from datetime import UTC, datetime

from llm_grounding.domain.answers import Assessment, JsonValue
from llm_grounding.domain.runs import RagRequest, RagRun
from llm_grounding.orchestration.architecture import Architecture
from llm_grounding.orchestration.context import RunContext
from llm_grounding.reasoning.models import ReasoningRequest
from llm_grounding.retrieval.models import RetrievalRequest


class RunFailedError(RuntimeError):
    """Expose the failed run and preserve the originating exception as its cause."""

    def __init__(self, run: RagRun) -> None:
        """Retain partial results without exposing external exception bodies."""
        super().__init__(run.error)
        self.run = run


class ArchitectureRunner:
    """Execute configured stages sequentially with a fresh context for each run."""

    def __init__(self, architecture: Architecture) -> None:
        """Inject complete composition without exposing it to individual components."""
        self._architecture = architecture

    async def run(
        self, request: RagRequest, metadata: dict[str, JsonValue] | None = None
    ) -> RagRun:
        """Execute all stages and return a completed run.

        Args:
            request: Semantic task and source identity.
            metadata: Credential-free reproducibility information supplied by the app.

        Raises:
            RunFailedError: If a component fails; its run retains completed stages.
        """
        architecture = self._architecture
        context = RunContext(
            architecture.name, architecture.version, metadata=dict(metadata or {})
        )
        run = RagRun(
            run_id=context.run_id,
            architecture_name=context.architecture_name,
            architecture_version=context.architecture_version,
            request=request,
            started_at=context.started_at,
            trace=context.trace,
            metadata=context.metadata,
        )

        async def execute[StageT](stage: str, operation: Awaitable[StageT]) -> StageT:
            """Trace a component boundary and preserve a sanitized failure artifact."""
            context.record(stage, "started")
            try:
                result = await operation
            except Exception as exc:
                # This orchestration boundary retains partial runs from arbitrary
                # implementations; it re-raises with context instead of suppressing.
                context.record(stage, "failed")
                run.status = "failed"
                run.completed_at = datetime.now(UTC)
                run.error = f"Stage {stage!r} failed ({type(exc).__name__})."
                raise RunFailedError(run) from exc
            context.record(stage, "completed")
            return result

        parsed = await execute("parse", architecture.parser.parse(request.source))
        run.metadata["source_sha256"] = parsed.metadata.get("sha256")
        run.metadata["page_count"] = parsed.page_count
        knowledge = await execute(
            "build_knowledge", architecture.knowledge_builder.build(parsed)
        )
        retrieved = await execute(
            "retrieve",
            architecture.retriever.retrieve(RetrievalRequest(request.task, knowledge)),
        )
        run.retrieval_output = retrieved.results
        run.retrieval_history.append(retrieved.results)
        ranked = await execute("rerank", architecture.reranker.rerank(retrieved))
        run.metadata["reasoning_evidence_ids"] = [
            result.evidence.evidence_id for result in ranked.results
        ]
        reasoning = await execute(
            "reason",
            architecture.reasoner.reason(
                ReasoningRequest(request.task, ranked.results)
            ),
        )
        run.reasoning_output = reasoning
        assessments: list[Assessment] = []
        for index, assessor in enumerate(architecture.assessors):
            assessments.extend(
                await execute(
                    f"assess:{index}", assessor.assess(reasoning, ranked.results)
                )
            )
            run.assessments = tuple(assessments)
        decision = await execute(
            "decide", architecture.decision_policy.decide(reasoning, run.assessments)
        )
        run.decision = decision
        run.answer = await execute("render", architecture.renderer.render(decision))
        run.status = "completed"
        run.completed_at = datetime.now(UTC)
        return run
