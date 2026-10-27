"""Current grounded query orchestration, separate from document ingestion."""

import json
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, cast
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
from llm_grounding.domain.evidence import EvidenceType, EvidenceUnit, RetrievalResult
from llm_grounding.embedding.factory import build_embedder
from llm_grounding.embedding.protocol import Embedder
from llm_grounding.infrastructure.config.models import RerankerConfig
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


class BasicRagRunner:
    """Run project-scoped retrieval, reasoning, validation, and rendering."""

    def __init__(
        self,
        vector: Retriever,
        summary: Retriever,
        reasoner: BasicGroundedReasoner,
        assessor: SourceReferenceAssessor,
        policy: BasicGroundingPolicy,
        renderer: GroundedAnswerRenderer,
        diagnostics: GroundingDiagnostics | None = None,
        embedding_config: EmbeddingConfig | None = None,
        reranker: Reranker | None = None,
        reranker_config: RerankerConfig | None = None,
        reranker_builder: Callable[[RerankerConfig], Reranker] | None = None,
        repository: PostgresRepository | None = None,
    ) -> None:
        """Inject independently testable query stages."""
        self._vector, self._summary, self._reasoner = vector, summary, reasoner
        self._assessor, self._policy, self._renderer = assessor, policy, renderer
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
        self._last_retrieval_response: RetrievalResponse | None = None

    async def run(
        self,
        request: CorpusRequest,
        history: tuple[Message, ...] = (),
        *,
        generate_chat_name: bool = False,
        runtime_config: dict[str, Any] | None = None,
    ) -> tuple[str, Answer, dict[str, object]]:
        """Execute Basic RAG and retain metadata suitable for persistence."""
        traces = TraceRecorder()
        usage = UsageAccumulator()
        total_span = traces.start("rag_e2e")
        values = self._runtime_values(runtime_config)
        retrieval_span = traces.start("retrieval")
        candidates = await self._retrieve_candidates(request, values)
        traces.end(retrieval_span, output_count=len(candidates))
        rerank_span = traces.start("rerank", input_count=len(candidates))
        chunks = await self._rerank_chunks(request.task, candidates, values)
        usage.add_reranker()
        traces.end(rerank_span, output_count=len(chunks))
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
                    "retrieval_stages": [
                        stage.model_dump(mode="json")
                        for stage in build_retrieval_stages((), run_id=str(uuid4()))
                    ],
                    "stage_graph": [
                        stage.model_dump(mode="json")
                        for stage in build_retrieval_stage_graph(
                            (), run_id=str(uuid4())
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
        reasoning = await self._reasoner.reason(context, **reasoner_kwargs)
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
                "stage_graph": [
                    stage.model_dump(mode="json")
                    for stage in build_retrieval_stage_graph(
                        chunks, run_id=reasoning.result_id
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
            },
        )

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
        self, request: CorpusRequest, values: dict[str, Any]
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
            if isinstance(self._summary, SummaryRetriever):
                response = await self._summary.retrieve(
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
                response = await self._summary.retrieve(retrieval_request, context)
        else:
            response = await self._vector.retrieve(retrieval_request, context)
        self._last_retrieval_response = response
        return response.results

    async def _rerank_chunks(
        self,
        query: str,
        candidates: tuple[ChunkRetrievalResult, ...],
        values: dict[str, Any],
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
        reranker = self._reranker_for(values)
        ranked = await reranker.rerank(response, top_k=rerank_top_k)
        by_id = {result.chunk.chunk_id: result for result in candidates}
        try:
            return tuple(
                ChunkRetrievalResult(
                    by_id[result.evidence.evidence_id].chunk,
                    result.rank,
                    result.retriever_id,
                    result.score,
                )
                for result in ranked.results
            )
        except KeyError as exc:
            raise RuntimeError(
                "Reranker returned an unknown retrieval identifier"
            ) from exc

    def _reranker_for(self, values: dict[str, Any]) -> Reranker:
        """Resolve and cache a request-local reranker when evaluation overrides it."""
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
