"""Current grounded query orchestration, separate from document ingestion."""

import json
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, cast
from uuid import uuid4

from llm_grounding.assessment.protocol import Assessor
from llm_grounding.decision.protocol import DecisionPolicy
from llm_grounding.domain.answers import (
    Answer,
    Assessment,
    DecisionAction,
    DecisionResult,
)
from llm_grounding.domain.corpus import (
    ChunkRetrievalResult,
    CorpusRequest,
    GroundingContext,
    Message,
    TaskType,
)
from llm_grounding.domain.evidence import EvidenceType, EvidenceUnit, RetrievalResult
from llm_grounding.embedding.factory import build_embedder
from llm_grounding.embedding.protocol import Embedder
from llm_grounding.infrastructure.config.models import (
    ExecutionBranchConfig,
    PipelineExecutionConfig,
    RerankerConfig,
)
from llm_grounding.infrastructure.database.repositories import PostgresRepository
from llm_grounding.ingestion.models import EmbeddingConfig
from llm_grounding.observability.embedding import embedding_diagnostics
from llm_grounding.observability.models import LLMUsage, RagUsage
from llm_grounding.observability.trace import TraceRecorder
from llm_grounding.observability.usage import UsageAccumulator
from llm_grounding.orchestration.grounding_diagnostics import GroundingDiagnostics
from llm_grounding.orchestration.retrieval_stages import (
    build_retrieval_stage_graph,
    build_retrieval_stages,
)
from llm_grounding.reasoning.basic_grounded import BasicGroundedReasoner
from llm_grounding.rendering.grounded_answer import GroundedAnswerRenderer
from llm_grounding.retrieval.models import (
    RerankResponse,
    RetrievalContext,
    RetrievalRequest,
    RetrievalResponse,
)
from llm_grounding.retrieval.protocol import Retriever
from llm_grounding.retrieval.reranking.identity import IdentityReranker
from llm_grounding.retrieval.reranking.protocol import Reranker
from llm_grounding.retrieval.vector import SummaryRetriever


def _artifact_tuple(
    artifacts: dict[str, object] | None, key: str
) -> tuple[ChunkRetrievalResult, ...] | None:
    """Read one typed in-memory output from an alternative source path."""
    if artifacts is None:
        return None
    value = artifacts.get(key)
    if not isinstance(value, tuple) or not all(
        isinstance(item, ChunkRetrievalResult) for item in value
    ):
        return None
    return value


def _alternative_quality(
    metadata: Mapping[str, object],
) -> tuple[float, int, int, int]:
    """Rank unaccepted candidates using measurements already produced by stages."""
    raw_scores = metadata.get("assessment_scores")
    scores = (
        [
            float(score)
            for score in raw_scores
            if isinstance(score, (int, float)) and not isinstance(score, bool)
        ]
        if isinstance(raw_scores, list)
        else []
    )
    action = metadata.get("decision")
    action_value = action.value if isinstance(action, DecisionAction) else action
    decision_rank = {
        "accept": 3,
        "abstain": 2,
        "retrieve_again": 2,
        "merge_branches": 2,
        "select_branches": 1,
        "reject": 0,
    }.get(action_value, 0)
    evidence_ids = metadata.get("retrieved_chunk_ids")
    evidence_count = len(evidence_ids) if isinstance(evidence_ids, list) else 0
    return (
        max(scores, default=-1.0),
        sum(score >= 0.5 for score in scores),
        decision_rank,
        evidence_count,
    )


class BasicRagRunner:
    """Run project-scoped retrieval, reasoning, validation, and rendering."""

    def __init__(
        self,
        vector: Retriever,
        summary: Retriever,
        reasoner: BasicGroundedReasoner,
        assessor: Assessor | Sequence[Assessor],
        policy: DecisionPolicy,
        renderer: GroundedAnswerRenderer,
        diagnostics: GroundingDiagnostics | None = None,
        embedding_config: EmbeddingConfig | None = None,
        reranker: Reranker | None = None,
        reranker_config: RerankerConfig | None = None,
        reranker_builder: Callable[[RerankerConfig], Reranker] | None = None,
        repository: PostgresRepository | None = None,
        execution: PipelineExecutionConfig | None = None,
        branch_retrievers: Mapping[str, Retriever] | None = None,
        branch_rerankers: Mapping[str, Reranker] | None = None,
        branch_assessors: Mapping[str, Sequence[Assessor]] | None = None,
        alternative_retrievers: Mapping[str, Retriever] | None = None,
        alternative_summaries: Mapping[str, Retriever] | None = None,
        alternative_rerankers: Mapping[str, Reranker] | None = None,
        alternative_assessors: Mapping[str, Sequence[Assessor]] | None = None,
        alternative_reasoners: Mapping[str, BasicGroundedReasoner] | None = None,
        alternative_renderers: Mapping[str, GroundedAnswerRenderer] | None = None,
    ) -> None:
        """Inject independently testable query stages."""
        self._vector, self._summary, self._reasoner = vector, summary, reasoner
        self._assessors = (
            tuple(assessor)
            if isinstance(assessor, (tuple, list))
            else (assessor,)
        )
        self._policy, self._renderer = policy, renderer
        self._execution = execution or PipelineExecutionConfig()
        self._reranker = reranker or IdentityReranker()
        self._reranker_config = reranker_config or RerankerConfig()
        if reranker_builder is None:
            from llm_grounding.orchestration.registry import ComponentRegistry

            reranker_builder = ComponentRegistry().build_reranker
        self._reranker_builder = reranker_builder
        self._reranker_cache = {
            self._reranker_config.model_dump_json(): self._reranker
        }
        self._diagnostics = diagnostics or GroundingDiagnostics()
        self._embedding_config = embedding_config or getattr(
            vector, "embedding_config", None
        )
        default_embedder = getattr(vector, "default_embedder", None)
        self._embedders: dict[str, Embedder] = {}
        if default_embedder is not None:
            self._embedders[
                (
                    self._embedding_config.identity()
                    if self._embedding_config is not None
                    else default_embedder.model_identity
                )
            ] = default_embedder
        self._repository = repository
        self._branch_retrievers = dict(branch_retrievers or {})
        self._branch_rerankers = dict(branch_rerankers or {})
        self._branch_assessors = {
            key: tuple(value) for key, value in (branch_assessors or {}).items()
        }
        self._alternative_retrievers = dict(alternative_retrievers or {})
        self._alternative_summaries = dict(alternative_summaries or {})
        self._alternative_rerankers = dict(alternative_rerankers or {})
        self._alternative_assessors = {
            key: tuple(value) for key, value in (alternative_assessors or {}).items()
        }
        self._alternative_reasoners = dict(alternative_reasoners or {})
        self._alternative_renderers = dict(alternative_renderers or {})
        self._last_retrieval_response: RetrievalResponse | None = None

    async def run(
        self,
        request: CorpusRequest,
        history: tuple[Message, ...] = (),
        *,
        generate_chat_name: bool = False,
        runtime_config: dict[str, Any] | None = None,
    ) -> tuple[str, Answer, dict[str, object]]:
        """Execute the configured topology with bounded branch or retry control."""
        if self._execution.alternatives:
            return await self._run_alternatives(
                request,
                history,
                generate_chat_name=generate_chat_name,
                runtime_config=runtime_config,
            )
        if self._execution.strategy == "linear":
            return self._public_result(await self._run_once(
                request,
                history,
                generate_chat_name=generate_chat_name,
                runtime_config=runtime_config,
            ))
        if self._execution.strategy == "branching":
            return self._public_result(await self._run_branching(
                request,
                history,
                generate_chat_name=generate_chat_name,
                runtime_config=runtime_config,
            ))
        return self._public_result(await self._run_recursive(
            request,
            history,
            generate_chat_name=generate_chat_name,
            runtime_config=runtime_config,
        ))

    async def _run_once(
        self,
        request: CorpusRequest,
        history: tuple[Message, ...] = (),
        *,
        generate_chat_name: bool = False,
        runtime_config: dict[str, Any] | None = None,
        _candidate_override: tuple[ChunkRetrievalResult, ...] | None = None,
        _chunk_override: tuple[ChunkRetrievalResult, ...] | None = None,
        _policy_override: DecisionPolicy | None = None,
        _retriever_override: Retriever | None = None,
        _summary_override: Retriever | None = None,
        _reranker_override: Reranker | None = None,
        _assessor_override: Sequence[Assessor] | None = None,
        _reasoner_override: BasicGroundedReasoner | None = None,
        _renderer_override: GroundedAnswerRenderer | None = None,
    ) -> tuple[str, Answer, dict[str, object]]:
        """Execute one linear pass and retain metadata suitable for persistence."""
        traces = TraceRecorder()
        usage = UsageAccumulator()
        total_span = traces.start("rag_e2e")
        values = self._runtime_values(runtime_config)
        policy = _policy_override or self._policy
        policy_token_budget = getattr(policy, "token_budget", None)
        if "context_budget_tokens" not in values and isinstance(
            policy_token_budget, int
        ):
            values["context_budget_tokens"] = policy_token_budget
        policy_top_k = getattr(policy, "top_k", None)
        if "context_top_k" not in values and isinstance(policy_top_k, int):
            values["context_top_k"] = policy_top_k
        retrieval_span = traces.start("retrieval")
        retrieval_performed = _candidate_override is None and _chunk_override is None
        if _candidate_override is not None:
            candidates = _candidate_override
        elif _chunk_override is not None:
            candidates = _chunk_override
        else:
            candidates = await self._retrieve_candidates(
                request,
                values,
                retriever_override=_retriever_override,
                summary_override=_summary_override,
            )
        if retrieval_performed:
            usage.add_retrieval()
        traces.end(retrieval_span, output_count=len(candidates))
        rerank_span = traces.start("rerank", input_count=len(candidates))
        reranked_chunks = (
            _chunk_override
            if _chunk_override is not None
            else await self._rerank_chunks(
                request.task,
                candidates,
                values,
                reranker_override=_reranker_override,
            )
        )
        if _chunk_override is None:
            usage.add_reranker()
        traces.end(rerank_span, output_count=len(reranked_chunks))
        selection_span = traces.start(
            "context_selection", input_count=len(reranked_chunks)
        )
        chunks = self._select_context(reranked_chunks, values)
        traces.end(selection_span, output_count=len(chunks))
        query_embedding = (
            self._last_retrieval_response.query_embedding
            if self._last_retrieval_response is not None
            else getattr(self._vector, "last_query_embedding", None)
        )
        if query_embedding is not None:
            usage.add_embedding()
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
                    "observation_status": "abstained",
                    "retrieved_chunk_ids": [],
                    "retrieval_results": [],
                    "_artifacts": {"candidates": candidates, "chunks": chunks},
                    "retrieval_stages": [
                        stage.model_dump(mode="json")
                        for stage in build_retrieval_stages(
                            candidates,
                            run_id=str(uuid4()),
                            reranked_context=(
                                reranked_chunks if _chunk_override is None else None
                            ),
                            final_context=(),
                        )
                    ],
                    "stage_graph": [
                        stage.model_dump(mode="json")
                        for stage in build_retrieval_stage_graph(
                            candidates,
                            run_id=str(uuid4()),
                            reranked_context=(
                                reranked_chunks if _chunk_override is None else None
                            ),
                            final_context=(),
                        )
                    ],
                    "trace": [span.model_dump(mode="json") for span in traces.spans],
                    "usage": self._usage_with_trace(usage, traces).model_dump(
                        mode="json"
                    ),
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
                    "query_embedding_configuration": self._embedding_metadata(),
                    "retrieval_metadata": (
                        self._last_retrieval_response.metadata
                        if self._last_retrieval_response is not None
                        else {}
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
        reasoner_kwargs = {
            key: values[key]
            for key in ("prompt_id", "model_name")
            if isinstance(values.get(key), str)
        }
        reasoning = await (_reasoner_override or self._reasoner).reason(
            context, **reasoner_kwargs
        )
        generation_data = reasoning.metadata.get("generation")
        if isinstance(generation_data, dict):
            generation_usage = generation_data.get("usage")
            if isinstance(generation_usage, dict):
                usage.add_llm(LLMUsage.model_validate(generation_usage))
        traces.end(reasoning_span, output_count=1)
        assessment_span = traces.start("assessment")
        assessment_values: list[Assessment] = []
        assessors = self._branch_assessors.get(
            str(values.get("branch_id")),
            tuple(_assessor_override)
            if _assessor_override is not None
            else self._assessors,
        )
        for assessor in assessors:
            assessment_values.extend(
                await assessor.assess_context(reasoning, context)
            )
        assessments = tuple(assessment_values)
        traces.end(assessment_span, output_count=len(assessments))
        decision_span = traces.start("decision", input_count=len(assessments))
        policy_result = reasoning
        if hasattr(policy, "token_budget"):
            policy_result = reasoning.__class__(
                reasoning.result_id,
                reasoning.payload,
                reasoning.claims,
                {
                    **reasoning.metadata,
                    "context_token_count": sum(
                        len(item.chunk.text.split()) for item in chunks
                    ),
                },
            )
        decision = await policy.decide(policy_result, assessments)
        traces.end(decision_span, output_count=1)
        self._diagnostics.report(context, reasoning, assessments, decision)
        rejected = decision.action == DecisionAction.REJECT
        render_span = traces.start("render")
        answer = (
            self._rejected_answer(decision)
            if rejected
            else await (_renderer_override or self._renderer).render_context(
                decision, context
            )
        )
        traces.end(render_span, output_count=1)
        traces.end(total_span, output_count=1)
        return (
            reasoning.result_id,
            answer,
            {
                "status": ("rejected" if rejected else "completed"),
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
                        "retrieval_score": (
                            item.ranking_score.__dict__
                            if item.ranking_score is not None
                            else None
                        ),
                        "rerank_score": (
                            item.rerank_score.__dict__
                            if item.rerank_score is not None
                            else None
                        ),
                    }
                    for item in chunks
                ],
                "assessment_labels": [item.label for item in assessments],
                "assessment_scores": [item.score for item in assessments],
                "assessment_dimensions": [item.dimension for item in assessments],
                "decision": decision.action,
                "decision_record": {
                    "action": decision.action,
                    "metadata": decision.metadata,
                },
                "assessments": [
                    {
                        "assessor": item.assessor,
                        "dimension": item.dimension,
                        "label": item.label,
                        "score": item.score,
                        "status": item.status,
                        "evidence_ids": list(item.evidence_ids),
                        "metadata": item.metadata,
                    }
                    for item in assessments
                ],
                "decision_metadata": decision.metadata,
                "parameter_overrides": decision.metadata.get("parameter_overrides", {}),
                "branch_ids": decision.metadata.get("branch_ids", []),
                "generated_chat_name": reasoning.payload.get("chat_name"),
                "generation": reasoning.metadata.get("generation", {}),
                "prompt_id": reasoning.metadata.get("prompt_id"),
                "prompt_version": reasoning.metadata.get("prompt_version"),
                "answer_payload": answer.payload,
                "evidence_ids": list(answer.evidence_ids),
                "answer_metadata": answer.metadata,
                "retrieval_stages": [
                    stage.model_dump(mode="json")
                    for stage in build_retrieval_stages(
                        candidates,
                        run_id=reasoning.result_id,
                        reranked_context=(
                            reranked_chunks if _chunk_override is None else None
                        ),
                        final_context=chunks,
                    )
                ],
                "stage_graph": [
                    stage.model_dump(mode="json")
                    for stage in build_retrieval_stage_graph(
                        candidates,
                        run_id=reasoning.result_id,
                        reranked_context=(
                            reranked_chunks if _chunk_override is None else None
                        ),
                        final_context=chunks,
                    )
                ],
                "trace": [span.model_dump(mode="json") for span in traces.spans],
                "usage": self._usage_with_trace(usage, traces).model_dump(mode="json"),
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
                "query_embedding_configuration": self._embedding_metadata(),
                "retrieval_metadata": (
                    self._last_retrieval_response.metadata
                    if self._last_retrieval_response is not None
                    else {}
                ),
                "created_at": datetime.now(UTC).isoformat(),
                "_artifacts": {"candidates": candidates, "chunks": chunks},
            },
        )

    async def _run_branching(
        self,
        request: CorpusRequest,
        history: tuple[Message, ...],
        *,
        generate_chat_name: bool,
        runtime_config: dict[str, Any] | None,
    ) -> tuple[str, Answer, dict[str, object]]:
        """Execute configured branches sequentially and join their outcomes."""
        return await self._run_branch_group(
            self._execution.branches,
            request,
            history,
            generate_chat_name=generate_chat_name,
            runtime_config=runtime_config,
            evidence_merge=self._execution.evidence_merge,
        )

    async def _run_branch_group(
        self,
        branches: Sequence[ExecutionBranchConfig],
        request: CorpusRequest,
        history: tuple[Message, ...],
        *,
        generate_chat_name: bool,
        runtime_config: dict[str, Any] | None,
        source_candidates: tuple[ChunkRetrievalResult, ...] | None = None,
        source_chunks: tuple[ChunkRetrievalResult, ...] | None = None,
        policy_override: DecisionPolicy | None = None,
        evidence_merge: str = "replace",
        alternative_id: str | None = None,
    ) -> tuple[str, Answer, dict[str, object]]:
        """Execute branches with optional outputs reused from an earlier path."""
        if len(branches) > self._execution.max_stage_calls:
            raise RuntimeError("Branch count exceeds the configured execution budget")
        base_values = dict(runtime_config or {})
        shared_candidates = source_candidates
        shared_chunks = source_chunks
        alternative_retriever = self._alternative_retrievers.get(alternative_id or "")
        alternative_summary = self._alternative_summaries.get(alternative_id or "")
        alternative_reranker = self._alternative_rerankers.get(alternative_id or "")
        alternative_assessors = self._alternative_assessors.get(alternative_id or "")
        alternative_reasoner = self._alternative_reasoners.get(alternative_id or "")
        alternative_renderer = self._alternative_renderers.get(alternative_id or "")
        if any(branch.start_at != "retriever" for branch in branches):
            if shared_candidates is None:
                shared_candidates = await self._retrieve_candidates(
                    request,
                    base_values,
                    retriever_override=alternative_retriever,
                    summary_override=alternative_summary,
                )
            if shared_chunks is None:
                shared_chunks = await self._rerank_chunks(
                    request.task,
                    shared_candidates,
                    base_values,
                    reranker_override=alternative_reranker,
                )
        outcomes: list[tuple[str, tuple[str, Answer, dict[str, object]]]] = []
        for branch in branches:
            branch_values = dict(base_values)
            branch_values.update(branch.parameter_overrides)
            branch_values["branch_id"] = (
                f"{alternative_id}:{branch.id}" if alternative_id else branch.id
            )
            if branch.retriever is not None:
                branch_values["retriever"] = branch.retriever
            if branch.reranker is not None:
                branch_values["reranker"] = branch.reranker
            result = await self._run_once(
                request,
                history,
                generate_chat_name=generate_chat_name,
                runtime_config=branch_values,
                _candidate_override=(
                    shared_candidates if branch.start_at == "reranker" else None
                ),
                _chunk_override=(
                    shared_chunks if branch.start_at == "assessor" else None
                ),
                _policy_override=policy_override,
                _retriever_override=(
                    self._branch_retrievers.get(branch.retriever, alternative_retriever)
                    if branch.retriever is not None
                    else alternative_retriever
                ),
                _summary_override=alternative_summary,
                _reranker_override=(
                    self._branch_rerankers.get(branch.reranker, alternative_reranker)
                    if branch.reranker is not None
                    else alternative_reranker
                ),
                _assessor_override=alternative_assessors,
                _reasoner_override=alternative_reasoner,
                _renderer_override=alternative_renderer,
            )
            outcomes.append((branch.id, result))
        return self._join_branch_outcomes(outcomes, evidence_merge=evidence_merge)

    async def _run_alternatives(
        self,
        request: CorpusRequest,
        history: tuple[Message, ...],
        *,
        generate_chat_name: bool,
        runtime_config: dict[str, Any] | None,
    ) -> tuple[str, Answer, dict[str, object]]:
        """Execute ordered alternatives using explicit in-memory path outputs."""
        alternatives = self._execution.alternatives
        attempts: list[dict[str, object]] = []
        path_outputs: dict[tuple[str, str], dict[str, object]] = {}
        results: list[tuple[str, Answer, dict[str, object]]] = []
        accepted = False
        for attempt_number, alternative in enumerate(
            alternatives[: self._execution.max_attempts], start=1
        ):
            values = dict(runtime_config or {})
            for stage, overrides in alternative.stage_overrides.items():
                if not isinstance(overrides, Mapping):
                    continue
                immutable = {
                    "retriever": {"type", "implementation", "requirements"},
                    "reranker": {"type", "implementation", "model_id"},
                    "reasoner": {"type", "implementation", "model_ref"},
                }.get(stage, set())
                values.update(
                    {
                        key: value
                        for key, value in overrides.items()
                        if key not in immutable
                    }
                )
            source_artifacts: dict[str, object] | None = None
            if alternative.source is not None:
                source_artifacts = path_outputs.get(
                    (alternative.source.alternative_id, alternative.source.path_id)
                )
                if source_artifacts is None:
                    raise RuntimeError(
                        "Alternative source output is unavailable: "
                        f"{alternative.source.alternative_id}/{alternative.source.path_id}"
                    )
            source_candidates = (
                _artifact_tuple(source_artifacts, "candidates")
                if source_artifacts is not None
                and alternative.source is not None
                and alternative.source.restart_stage in {"reranker", "assessor"}
                else None
            )
            source_chunks = (
                _artifact_tuple(source_artifacts, "chunks")
                if source_artifacts is not None
                and alternative.source is not None
                and alternative.source.restart_stage == "assessor"
                else None
            )
            if alternative.branches:
                result = await self._run_branch_group(
                    alternative.branches,
                    request,
                    history,
                    generate_chat_name=generate_chat_name,
                    runtime_config=values,
                    source_candidates=source_candidates,
                    source_chunks=source_chunks,
                    evidence_merge=self._execution.evidence_merge,
                    alternative_id=alternative.id,
                )
            else:
                result = await self._run_once(
                    request,
                    history,
                    generate_chat_name=generate_chat_name,
                    runtime_config=values,
                    _candidate_override=source_candidates,
                    _chunk_override=source_chunks,
                    _retriever_override=self._alternative_retrievers.get(
                        alternative.id
                    ),
                    _summary_override=self._alternative_summaries.get(alternative.id),
                    _reranker_override=self._alternative_rerankers.get(alternative.id),
                    _assessor_override=self._alternative_assessors.get(alternative.id),
                    _reasoner_override=self._alternative_reasoners.get(alternative.id),
                    _renderer_override=self._alternative_renderers.get(alternative.id),
                )
            results.append(result)
            metadata = result[2]
            artifacts = metadata.get("_artifacts")
            if isinstance(artifacts, dict):
                path_outputs[(alternative.id, "primary")] = artifacts
            branch_outputs = metadata.get("_alternative_paths")
            if isinstance(branch_outputs, dict):
                for path_id, path_artifacts in branch_outputs.items():
                    if isinstance(path_id, str) and isinstance(path_artifacts, dict):
                        path_outputs[(alternative.id, path_id)] = path_artifacts
            accepted = (
                self._action_value(metadata.get("decision"))
                == DecisionAction.ACCEPT
            )
            attempts.append(
                {
                    "alternative_id": alternative.id,
                    "attempt": attempt_number,
                    "accepted": accepted,
                    "quality": _alternative_quality(metadata),
                    "decision": self._action_value(metadata.get("decision")),
                    "retrieved_chunk_ids": metadata.get("retrieved_chunk_ids", []),
                    "retrieval_stages": metadata.get("retrieval_stages", []),
                    "stage_graph": metadata.get("stage_graph", []),
                    "trace": metadata.get("trace", []),
                    "assessments": metadata.get("assessments", []),
                    "usage": metadata.get("usage", {}),
                }
            )
            if accepted:
                break
        if not results:
            raise RuntimeError("Unified execution requires at least one alternative")
        selected_index = len(results) - 1
        if not accepted and self._execution.final_failure_policy != "last_result":
            selected_index = max(
                range(len(results)),
                key=lambda index: (_alternative_quality(results[index][2]), index),
            )
        result = results[selected_index]
        result_id, answer, metadata = result
        if not accepted and self._execution.final_failure_policy in {
            "reject",
            "abstain",
        }:
            action = self._execution.final_failure_policy
            answer = Answer(
                "Insufficient evidence after all configured query alternatives.",
                payload=answer.payload,
                claims=answer.claims,
                evidence_ids=answer.evidence_ids,
                metadata={**answer.metadata, "final_failure_policy": action},
            )
        merged_metadata = dict(metadata)
        merged_metadata.update(
            {
                "execution_strategy": "alternatives",
                "alternative_id": attempts[selected_index]["alternative_id"],
                "execution_attempt": selected_index + 1,
                "execution_attempts": attempts,
                "termination_reason": (
                    "alternative_accepted" if accepted else "max_attempts_reached"
                ),
            }
        )
        merged_metadata.update(self._combined_attempt_observation_metadata(attempts))
        return self._public_result((result_id, answer, merged_metadata))

    def _join_branch_outcomes(
        self,
        outcomes: list[tuple[str, tuple[str, Answer, dict[str, object]]]],
        *,
        evidence_merge: str | None = None,
    ) -> tuple[str, Answer, dict[str, object]]:
        """Select or merge branch results while retaining every branch trace."""
        if not outcomes:
            raise RuntimeError("Branching execution requires at least one outcome")
        selected_ids: set[str] = set()
        for _, (_, _, metadata) in outcomes:
            raw_ids = metadata.get("branch_ids")
            if isinstance(raw_ids, list):
                selected_ids.update(
                    value for value in raw_ids if isinstance(value, str)
                )
        selected = [
            item for item in outcomes if not selected_ids or item[0] in selected_ids
        ]
        if not selected:
            selected = outcomes
        accepted = [
            item
            for item in selected
            if self._action_value(item[1][2].get("decision"))
            in {DecisionAction.ACCEPT.value, DecisionAction.MERGE_BRANCHES.value}
        ]
        chosen = accepted[0] if accepted else selected[0]
        result_id, answer, metadata = chosen[1]
        branch_records = [
            {
                "branch_id": branch_id,
                "result_id": result[0],
                "status": result[2].get("status"),
                "decision": self._action_value(result[2].get("decision")),
                "retrieved_chunk_ids": result[2].get("retrieved_chunk_ids", []),
                "retrieval_results": result[2].get("retrieval_results", []),
                "assessment_labels": result[2].get("assessment_labels", []),
                "assessments": result[2].get("assessments", []),
                "retrieval_stages": result[2].get("retrieval_stages", []),
                "stage_graph": result[2].get("stage_graph", []),
                "trace": result[2].get("trace", []),
                "usage": result[2].get("usage", {}),
            }
            for branch_id, result in outcomes
        ]
        merged_metadata = dict(metadata)
        merged_metadata.update(
            {
                "execution_strategy": "branching",
                "branch_ids": [branch_id for branch_id, _ in selected],
                "branch_results": branch_records,
                "termination_reason": "branch_joined_at_decision_policy",
                "_alternative_paths": {
                    branch_id: result[2].get("_artifacts", {})
                    for branch_id, result in outcomes
                },
                "retrieval_stages": [
                    stage
                    for _, result in outcomes
                    for stage in result[2].get("retrieval_stages", [])
                    if isinstance(stage, dict)
                ],
                "stage_graph": [
                    stage
                    for _, result in outcomes
                    for stage in result[2].get("stage_graph", [])
                    if isinstance(stage, dict)
                ],
                "trace": [
                    span
                    for _, result in outcomes
                    for span in result[2].get("trace", [])
                    if isinstance(span, dict)
                ],
                "assessments": [
                    assessment
                    for _, result in outcomes
                    for assessment in result[2].get("assessments", [])
                    if isinstance(assessment, dict)
                ],
                "usage": self._merge_usage(
                    [result[2].get("usage", {}) for _, result in outcomes]
                ),
            }
        )
        merge = evidence_merge or self._execution.evidence_merge
        if merge in {"union", "rrf"}:
            merged_metadata["retrieved_chunk_ids"] = self._merged_chunk_ids(outcomes)
            if merge == "rrf":
                merged_metadata["retrieval_results"] = self._fused_branch_results(
                    outcomes
                )
        return result_id, answer, merged_metadata

    async def _run_recursive(
        self,
        request: CorpusRequest,
        history: tuple[Message, ...],
        *,
        generate_chat_name: bool,
        runtime_config: dict[str, Any] | None,
    ) -> tuple[str, Answer, dict[str, object]]:
        """Execute bounded retrieval rounds until a policy terminates the run."""
        values = dict(runtime_config or {})
        attempts: list[dict[str, object]] = []
        results: list[tuple[str, Answer, dict[str, object]]] = []
        for round_number in range(self._execution.max_rounds):
            if (
                round_number > 0
                and round_number - 1 < len(self._execution.retry_overrides)
            ):
                values.update(self._execution.retry_overrides[round_number - 1])
            result = await self._run_once(
                request,
                history,
                generate_chat_name=generate_chat_name,
                runtime_config=values,
            )
            result_metadata = result[2]
            results.append(result)
            attempts.append(
                {
                    "round": round_number + 1,
                    "decision": self._action_value(result_metadata.get("decision")),
                    "retrieved_chunk_ids": result_metadata.get(
                        "retrieved_chunk_ids", []
                    ),
                    "retrieval_results": result_metadata.get("retrieval_results", []),
                    "trace": result_metadata.get("trace", []),
                    "retrieval_stages": result_metadata.get("retrieval_stages", []),
                    "stage_graph": result_metadata.get("stage_graph", []),
                    "assessments": result_metadata.get("assessments", []),
                    "usage": result_metadata.get("usage", {}),
                    "parameter_overrides": dict(values),
                }
            )
            action = self._action_value(result_metadata.get("decision"))
            if action not in self._execution.retry_on:
                break
            policy_overrides = result_metadata.get("parameter_overrides")
            if isinstance(policy_overrides, dict):
                values.update(policy_overrides)
        if not results:
            raise RuntimeError("Recursive execution did not produce an outcome")
        selected_index = len(results) - 1
        final_action = self._action_value(results[-1][2].get("decision"))
        if (
            final_action != DecisionAction.ACCEPT.value
            and self._execution.final_failure_policy != "last_result"
        ):
            selected_index = max(
                range(len(results)),
                key=lambda index: (_alternative_quality(results[index][2]), index),
            )
        result = results[selected_index]
        result_id, answer, metadata = result
        merged_metadata = dict(metadata)
        merged_metadata.update(
            {
                "execution_strategy": "recursive",
                "execution_round": len(attempts),
                "execution_attempts": attempts,
                "termination_reason": (
                    "max_rounds_reached"
                    if self._action_value(metadata.get("decision"))
                    in self._execution.retry_on
                    else "decision_policy_terminated"
                ),
            }
        )
        merged_metadata.update(self._combined_attempt_observation_metadata(attempts))
        if self._execution.evidence_merge in {"union", "rrf"}:
            merged_metadata["retrieved_chunk_ids"] = self._merged_attempt_chunk_ids(
                attempts
            )
            if self._execution.evidence_merge == "rrf":
                merged_metadata["retrieval_results"] = self._fused_attempt_results(
                    attempts
                )
        return result_id, answer, merged_metadata

    @staticmethod
    def _action_value(value: object) -> str | None:
        """Normalize enum and string decisions for persisted execution metadata."""
        if isinstance(value, DecisionAction):
            return value.value
        return value if isinstance(value, str) else None

    @staticmethod
    def _merge_usage(records: list[object]) -> dict[str, object]:
        """Aggregate usage records from independent branches or retry rounds."""
        values = [record for record in records if isinstance(record, dict)]
        if not values:
            return {}
        count_fields = (
            "llm_calls",
            "embedding_calls",
            "retrieval_calls",
            "reranker_calls",
        )
        token_fields = (
            "context_tokens",
            "total_input_tokens",
            "total_output_tokens",
            "total_tokens",
        )
        merged: dict[str, object] = {
            field: sum(
                int(value[field])
                for value in values
                if isinstance(value.get(field), int)
            )
            for field in count_fields
        }
        for field in token_fields:
            field_values = [value.get(field) for value in values]
            merged[field] = (
                sum(int(item) for item in field_values if isinstance(item, int))
                if field_values and all(isinstance(item, int) for item in field_values)
                else None
            )
        origins = {value.get("usage_origin") for value in values}
        merged["usage_origin"] = (
            "provider_reported" if origins == {"provider_reported"}
            else "unavailable"
            if not origins or origins == {"unavailable"}
            else "estimated"
        )
        merged["resource_usage"] = [
            resource
            for value in values
            for resource in value.get("resource_usage", [])
            if isinstance(resource, dict)
        ]
        merged["metadata"] = {
            "llm_usage": [
                item
                for value in values
                for item in (
                    value.get("metadata")
                    if isinstance(value.get("metadata"), dict)
                    else {}
                ).get("llm_usage", [])
                if isinstance(item, dict)
            ]
        }
        return merged

    @classmethod
    def _combined_attempt_observation_metadata(
        cls, attempts: list[dict[str, object]]
    ) -> dict[str, object]:
        """Expose every bounded retry attempt in the canonical observation."""
        return {
            "retrieval_stages": [
                stage
                for attempt in attempts
                for stage in attempt.get("retrieval_stages", [])
                if isinstance(stage, dict)
            ],
            "stage_graph": [
                stage
                for attempt in attempts
                for stage in attempt.get("stage_graph", [])
                if isinstance(stage, dict)
            ],
            "trace": [
                span
                for attempt in attempts
                for span in attempt.get("trace", [])
                if isinstance(span, dict)
            ],
            "assessments": [
                assessment
                for attempt in attempts
                for assessment in attempt.get("assessments", [])
                if isinstance(assessment, dict)
            ],
            "usage": cls._merge_usage(
                [attempt.get("usage", {}) for attempt in attempts]
            ),
        }

    @staticmethod
    def _public_result(
        result: tuple[str, Answer, dict[str, object]],
    ) -> tuple[str, Answer, dict[str, object]]:
        """Remove in-memory reuse artifacts before metadata persistence."""
        result_id, answer, metadata = result
        public_metadata = dict(metadata)
        public_metadata.pop("_artifacts", None)
        public_metadata.pop("_alternative_paths", None)
        return result_id, answer, public_metadata

    @staticmethod
    def _merged_chunk_ids(
        outcomes: list[tuple[str, tuple[str, Answer, dict[str, object]]]],
    ) -> list[str]:
        """Deduplicate branch evidence in deterministic branch and rank order."""
        merged: list[str] = []
        for _, (_, _, metadata) in outcomes:
            for value in metadata.get("retrieved_chunk_ids", []):
                if isinstance(value, str) and value not in merged:
                    merged.append(value)
        return merged

    @staticmethod
    def _fused_branch_results(
        outcomes: list[tuple[str, tuple[str, Answer, dict[str, object]]]],
    ) -> list[dict[str, object]]:
        """Add reciprocal-rank fusion scores without overwriting native scores."""
        fused: dict[str, dict[str, object]] = {}
        for _, (_, _, metadata) in outcomes:
            for item in metadata.get("retrieval_results", []):
                if not isinstance(item, dict) or not isinstance(
                    item.get("chunk_id"), str
                ):
                    continue
                chunk_id = item["chunk_id"]
                rank = item.get("rank")
                if not isinstance(rank, int):
                    continue
                record = fused.setdefault(chunk_id, dict(item))
                record["fusion_score"] = float(
                    record.get("fusion_score", 0.0)
                ) + 1.0 / (60 + rank)
        return sorted(
            fused.values(),
            key=lambda item: (
                -float(item.get("fusion_score", 0.0)),
                str(item["chunk_id"]),
            ),
        )

    @staticmethod
    def _merged_attempt_chunk_ids(attempts: list[dict[str, object]]) -> list[str]:
        """Deduplicate evidence across recursive rounds in execution order."""
        merged: list[str] = []
        for attempt in attempts:
            for value in attempt.get("retrieved_chunk_ids", []):
                if isinstance(value, str) and value not in merged:
                    merged.append(value)
        return merged

    @staticmethod
    def _fused_attempt_results(
        attempts: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        """Fuse recursive-round ranks without replacing each native score."""
        outcomes = [
            (
                str(attempt.get("round", "")),
                (
                    "",
                    Answer(""),
                    {
                        "retrieval_results": attempt.get("retrieval_results", [])
                    },
                ),
            )
            for attempt in attempts
        ]
        return BasicRagRunner._fused_branch_results(outcomes)

    async def retrieve(
        self,
        request: CorpusRequest,
        *,
        runtime_config: dict[str, Any] | None = None,
    ) -> tuple[ChunkRetrievalResult, ...]:
        """Retrieve and rerank the exact final context used by the query path."""
        values = self._runtime_values(runtime_config)
        candidates = await self._retrieve_candidates(request, values)
        return await self._rerank_chunks(request.task, candidates, values)

    async def _retrieve_candidates(
        self,
        request: CorpusRequest,
        values: dict[str, Any],
        *,
        retriever_override: Retriever | None = None,
        summary_override: Retriever | None = None,
    ) -> tuple[ChunkRetrievalResult, ...]:
        """Retrieve the candidate set before the configured reranker runs."""
        top_k = values.get("top_k") if isinstance(values.get("top_k"), int) else None
        if isinstance(values.get("retrieval_top_k"), int):
            top_k = values["retrieval_top_k"]
        embedder = self._request_embedder(values)
        retrieval_request = RetrievalRequest.from_corpus(request, top_k=top_k)
        context = RetrievalContext(
            repository=self._repository,
            embedder=embedder,
            embedding_config=self._embedding_config,
        )
        if request.task_type == TaskType.SUMMARY:
            summary = summary_override or self._branch_retrievers.get(
                str(values.get("retriever")), self._summary
            )
            if isinstance(summary, SummaryRetriever):
                response = await summary.retrieve(
                    retrieval_request,
                    context,
                    context_chars=(
                        values.get("context_chars")
                        if isinstance(values.get("context_chars"), int)
                        else None
                    ),
                    per_facet=(
                        values.get("summary_per_facet")
                        if isinstance(values.get("summary_per_facet"), int)
                        else None
                    ),
                )
            else:
                response = await summary.retrieve(retrieval_request, context)
        else:
            retriever = retriever_override or self._branch_retrievers.get(
                str(values.get("retriever")), self._vector
            )
            response = await retriever.retrieve(retrieval_request, context)
        self._last_retrieval_response = response
        return response.results

    async def _rerank_chunks(
        self,
        query: str,
        candidates: tuple[ChunkRetrievalResult, ...],
        values: dict[str, Any],
        *,
        reranker_override: Reranker | None = None,
    ) -> tuple[ChunkRetrievalResult, ...]:
        """Adapt persistent chunks to the shared reranker response contract."""
        if not candidates:
            return ()
        response = RerankResponse(
            query,
            tuple(
                RetrievalResult(
                    EvidenceUnit(
                        evidence_id=result.chunk.chunk_id,
                        document_id=result.chunk.document_id,
                        evidence_type=EvidenceType.TEXT,
                        source_uri="",
                        order=result.chunk.order,
                        page_number=result.chunk.page_start,
                        text=result.chunk.text,
                        metadata=result.chunk.metadata,
                    ),
                    result.rank,
                    result.retriever_id,
                    result.score,
                )
                for result in candidates
            ),
        )
        rerank_top_k = (
            values.get("rerank_top_k")
            if isinstance(values.get("rerank_top_k"), int)
            else None
        )
        reranker = reranker_override or self._reranker_for(values)
        ranked = await reranker.rerank(response, top_k=rerank_top_k)
        by_id = {result.chunk.chunk_id: result for result in candidates}
        try:
            return tuple(
                ChunkRetrievalResult(
                    by_id[result.evidence.evidence_id].chunk,
                    result.rank,
                    result.retriever_id,
                    result.score,
                    result.ranking_score,
                    result.rerank_score,
                )
                for result in ranked.results
            )
        except KeyError as exc:
            raise RuntimeError(
                "Reranker returned an unknown retrieval identifier"
            ) from exc

    @staticmethod
    def _select_context(
        chunks: tuple[ChunkRetrievalResult, ...], values: dict[str, Any]
    ) -> tuple[ChunkRetrievalResult, ...]:
        """Apply deterministic evidence filters and an optional token budget."""
        selected = chunks
        allowed = values.get("evidence_ids", values.get("chunk_ids"))
        if isinstance(allowed, (list, tuple, set)):
            allowed_ids = {value for value in allowed if isinstance(value, str)}
            selected = tuple(
                item for item in selected if item.chunk.chunk_id in allowed_ids
            )
        excluded = values.get("exclude_evidence_ids")
        if isinstance(excluded, (list, tuple, set)):
            excluded_ids = {value for value in excluded if isinstance(value, str)}
            selected = tuple(
                item for item in selected if item.chunk.chunk_id not in excluded_ids
            )
        context_top_k = values.get("context_top_k")
        if isinstance(context_top_k, int) and context_top_k > 0:
            selected = selected[:context_top_k]
        budget = values.get("context_budget_tokens")
        if not isinstance(budget, int) or budget < 1:
            return selected
        total = 0
        within_budget: list[ChunkRetrievalResult] = []
        for item in selected:
            token_count = len(item.chunk.text.split())
            if within_budget and total + token_count > budget:
                break
            if not within_budget and token_count > budget:
                continue
            within_budget.append(item)
            total += token_count
        return tuple(within_budget)

    def _reranker_for(self, values: dict[str, Any]) -> Reranker:
        """Resolve and cache a request-local reranker when evaluation overrides it."""
        requested_reranker = values.get("reranker")
        if isinstance(requested_reranker, str):
            branch_reranker = self._branch_rerankers.get(requested_reranker)
            if branch_reranker is not None:
                return branch_reranker
        override_keys = {
            "reranker",
            "rerank_top_k",
            "reranker_model_name",
            "reranker_device",
            "reranker_batch_size",
        }
        if not override_keys.intersection(values):
            return self._reranker
        settings = self._reranker_config.model_dump(mode="json")
        mapping = {
            "reranker": "type",
            "rerank_top_k": "rerank_top_k",
            "reranker_model_name": "model_name",
            "reranker_device": "device",
            "reranker_batch_size": "batch_size",
        }
        settings.update(
            {
                target: values[source]
                for source, target in mapping.items()
                if source in values
            }
        )
        if settings.get("implementation") in {"identity", "top_k"}:
            settings["model_id"] = None
        config = RerankerConfig.model_validate(settings)
        cache_key = json.dumps(settings, sort_keys=True, separators=(",", ":"))
        reranker = self._reranker_cache.get(cache_key)
        if reranker is None:
            reranker = self._reranker_builder(config)
            self._reranker_cache[cache_key] = reranker
        return reranker

    @staticmethod
    def _runtime_values(runtime_config: dict[str, Any] | None) -> dict[str, Any]:
        """Read the structured runtime configuration map."""
        values = dict(runtime_config or {})
        nested = values.get("config")
        return dict(nested) if isinstance(nested, dict) else values

    def _request_embedder(self, values: dict[str, Any]) -> Embedder | None:
        """Resolve and cache a request-specific embedding provider when requested."""
        requested = values.get("embedding", values.get("embedding_config"))
        if requested is None:
            return None
        base = self._embedding_config
        if isinstance(requested, EmbeddingConfig):
            config = requested
        elif isinstance(requested, dict):
            if base is None:
                raise ValueError(
                    "Embedding overrides require a configured base embedding"
                )
            config = EmbeddingConfig.model_validate(
                {**base.model_dump(mode="json"), **requested}
            )
        else:
            raise ValueError("embedding override must be an EmbeddingConfig or object")
        if base is not None and not config.is_compatible_with(base):
            raise ValueError(
                "Query embedding configuration is incompatible with the indexed "
                "embedding space"
            )
        identity = config.identity()
        embedder = self._embedders.get(identity)
        if embedder is None:
            embedder = cast(Embedder, build_embedder(config))
            self._embedders[identity] = embedder
        return embedder

    def _embedding_metadata(self) -> dict[str, object] | None:
        """Return the effective query embedding contract without credentials."""
        if self._embedding_config is None:
            return None
        return {
            "provider": self._embedding_config.provider,
            "model": self._embedding_config.model,
            "revision": self._embedding_config.revision,
            "embedding_space_id": self._embedding_config.embedding_space_id,
            "vector_kind": self._embedding_config.representation,
            "dimensions": self._embedding_config.dimension,
            "normalization": self._embedding_config.normalize,
            "similarity_metric": self._embedding_config.similarity_metric,
            "configuration_identity": self._embedding_config.identity(),
            "vector_identity": self._embedding_config.vector_identity(),
        }

    @staticmethod
    def _usage_with_trace(usage: UsageAccumulator, traces: TraceRecorder) -> RagUsage:
        """Include scoped local telemetry captured by completed execution spans."""
        for span in traces.spans:
            if span.resource_usage is not None:
                usage.add_resource_usage(span.resource_usage)
        return usage.build()

    @staticmethod
    def _rejected_answer(decision: DecisionResult) -> Answer:
        """Return a safe response without rendering rejected model content."""
        reason = str(decision.metadata.get("reason", "grounding_validation_failed"))
        return Answer(
            "Grounding failure: the generated response did not pass source validation.",
            metadata={"grounding_rejected": True, "reason": reason},
        )
