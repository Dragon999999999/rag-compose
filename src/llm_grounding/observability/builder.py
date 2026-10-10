"""Translate existing Basic RAG runtime objects into canonical observations."""

from datetime import UTC, datetime
from typing import Any

from llm_grounding.domain.answers import (
    Answer,
    Assessment,
    DecisionResult,
    ReasoningResult,
)
from llm_grounding.domain.corpus import ChunkRetrievalResult, CorpusRequest
from llm_grounding.domain.runs import TraceEvent
from llm_grounding.observability.models import (
    AssessmentModel,
    CitationModel,
    ClaimModel,
    DecisionModel,
    EffectiveConfiguration,
    EmbeddingDiagnostics,
    GenerationInfo,
    ObservationAnswer,
    ObservationRequest,
    RagRunObservation,
    RagUsage,
    StageSnapshot,
    TraceSpan,
)
from llm_grounding.observability.retrieved_items import retrieved_item
from llm_grounding.orchestration.retrieval_stages import build_retrieval_stages
from llm_grounding.reasoning.basic_grounded import GroundedResponsePayload


class ObservationBuilder:
    """Build the one serialization shape shared by APIs and diagnostics."""

    def build(
        self,
        *,
        run_id: str,
        request_id: str,
        corpus_id: str,
        architecture_name: str,
        architecture_version: int | str,
        request: CorpusRequest,
        chunks: tuple[ChunkRetrievalResult, ...] = (),
        history: tuple[Any, ...] = (),
        answer: Answer | None = None,
        reasoning: ReasoningResult | None = None,
        assessments: tuple[Assessment, ...] = (),
        decision: DecisionResult | None = None,
        retrieval_stages: list[Any] | None = None,
        stages: list[StageSnapshot] | None = None,
        trace: list[TraceSpan] | tuple[TraceSpan, ...] = (),
        legacy_trace: list[TraceEvent] | tuple[TraceEvent, ...] = (),
        usage: RagUsage | None = None,
        effective_configuration: EffectiveConfiguration | None = None,
        status: str = "completed",
        created_at: datetime | None = None,
        completed_at: datetime | None = None,
        error: str | None = None,
        generation: GenerationInfo | None = None,
        query_embedding_id: str | None = None,
        embedding: EmbeddingDiagnostics | None = None,
        conversation_id: str | None = None,
        turn_id: str | None = None,
        turn_index: int | None = None,
        conversation_state: Any | None = None,
        benchmark_case_id: str | None = None,
        experiment_id: str | None = None,
        seed: int | None = None,
        perturbation_metadata: dict[str, Any] | None = None,
    ) -> RagRunObservation:
        """Build an observation while preserving whatever partial state exists."""
        payload = self._payload(answer, reasoning)
        retrieval_snapshots = retrieval_stages or build_retrieval_stages(
            chunks, run_id=run_id
        )
        return RagRunObservation(
            run_id=run_id,
            request_id=request_id,
            target_run_id=run_id,
            corpus_id=corpus_id,
            architecture_name=architecture_name,
            architecture_version=architecture_version,
            status=status,
            request=ObservationRequest(
                query=request.task,
                history=[self._history_item(item) for item in history],
                conversation_history=[self._history_item(item) for item in history],
                included_history_turn_ids=[
                    str(item.id) for item in history if hasattr(item, "id")
                ],
                parameters={"task_type": request.task_type.value},
                filters={"document_ids": list(request.document_ids)},
            ),
            answer=self._answer(payload, chunks, status),
            generation=generation or GenerationInfo(),
            retrieval=list(retrieval_snapshots),
            stages=list(stages or []),
            query_embedding_id=query_embedding_id,
            embedding=embedding,
            assessments=[
                self._assessment(item, index)
                for index, item in enumerate(assessments, 1)
            ],
            decision=self._decision(decision, assessments),
            trace=list(trace) + self._legacy_trace(legacy_trace),
            usage=usage or RagUsage(),
            effective_configuration=effective_configuration,
            created_at=created_at or datetime.now(UTC),
            completed_at=completed_at,
            error=error,
            configuration_hash=(
                effective_configuration.architecture.config_hash
                if effective_configuration is not None
                else None
            ),
            conversation_id=conversation_id,
            turn_id=turn_id,
            turn_index=turn_index,
            conversation_state=conversation_state,
            benchmark_case_id=benchmark_case_id,
            experiment_id=experiment_id,
            seed=seed,
            perturbation_metadata=perturbation_metadata or {},
        )

    def from_basic_rag_metadata(
        self,
        *,
        run_id: str,
        request_id: str,
        corpus_id: str,
        architecture_name: str,
        architecture_version: int | str,
        request: CorpusRequest,
        chunks: tuple[ChunkRetrievalResult, ...],
        answer: Answer | None,
        metadata: dict[str, object],
        effective_configuration: EffectiveConfiguration | None = None,
        history: tuple[Any, ...] = (),
    ) -> RagRunObservation:
        """Build from the metadata returned by ``BasicRagRunner.run``."""
        generation_data = metadata.get("generation")
        generation = GenerationInfo(
            model_id=(
                str(generation_data.get("model_id"))
                if isinstance(generation_data, dict)
                and generation_data.get("model_id") is not None
                else (
                    effective_configuration.generation.model_id
                    if effective_configuration and effective_configuration.generation
                    else None
                )
            ),
            prompt_id=(
                str(metadata.get("prompt_id"))
                if metadata.get("prompt_id") is not None
                else (
                    effective_configuration.prompt.prompt_id
                    if effective_configuration and effective_configuration.prompt
                    else None
                )
            ),
            prompt_version=(
                str(metadata.get("prompt_version"))
                if metadata.get("prompt_version") is not None
                else (
                    effective_configuration.prompt.prompt_version
                    if effective_configuration and effective_configuration.prompt
                    else None
                )
            ),
        )
        trace_data = metadata.get("trace", [])
        trace = (
            [
                TraceSpan.model_validate(item)
                for item in trace_data
                if isinstance(item, dict)
            ]
            if isinstance(trace_data, list)
            else []
        )
        usage_data = metadata.get("usage")
        usage = (
            RagUsage.model_validate(usage_data)
            if isinstance(usage_data, dict)
            else None
        )
        embedding_data = metadata.get("embedding")
        embedding = (
            EmbeddingDiagnostics.model_validate(embedding_data)
            if isinstance(embedding_data, dict)
            else None
        )
        stage_data = metadata.get("stage_graph", [])
        stages = (
            [
                StageSnapshot.model_validate(item)
                for item in stage_data
                if isinstance(item, dict)
            ]
            if isinstance(stage_data, list)
            else []
        )
        return self.build(
            run_id=run_id,
            request_id=request_id,
            corpus_id=corpus_id,
            architecture_name=architecture_name,
            architecture_version=architecture_version,
            request=request,
            chunks=chunks,
            answer=answer,
            effective_configuration=effective_configuration,
            status=str(
                metadata.get("observation_status", metadata.get("status", "completed"))
            ),
            created_at=self._datetime(metadata.get("created_at")),
            history=history,
            trace=trace,
            generation=generation,
            query_embedding_id=(
                str(metadata["query_embedding_id"])
                if metadata.get("query_embedding_id") is not None
                else None
            ),
            embedding=embedding,
            usage=usage,
            stages=stages,
        )

    @staticmethod
    def _payload(
        answer: Answer | None, reasoning: ReasoningResult | None
    ) -> GroundedResponsePayload | None:
        """Validate canonical payload data when the architecture provides it."""
        candidate = (
            answer.payload
            if answer and answer.payload
            else reasoning.payload
            if reasoning
            else None
        )
        if not candidate:
            return None
        try:
            return GroundedResponsePayload.model_validate(candidate)
        except ValueError:
            return None

    @staticmethod
    def _history_item(item: Any) -> dict[str, Any]:
        """Serialize history records without requiring a specific persistence type."""
        if hasattr(item, "role") and hasattr(item, "content"):
            return {"role": item.role, "content": item.content}
        return {"value": str(item)}

    @staticmethod
    def _answer(
        payload: GroundedResponsePayload | None,
        chunks: tuple[ChunkRetrievalResult, ...],
        run_status: str,
    ) -> ObservationAnswer | None:
        """Map claims and source IDs while retaining unresolved references."""
        if payload is None:
            return None
        by_id = {item.chunk.chunk_id: retrieved_item(item) for item in chunks}
        claims: list[ClaimModel] = []
        citations: list[CitationModel] = []
        for claim_index, claim in enumerate(payload.claims, 1):
            claim_id = f"claim_{claim_index}"
            citation_ids: list[str] = []
            for citation_index, chunk_id in enumerate(claim.source_chunk_ids, 1):
                citation_id = f"citation_{claim_index}_{citation_index}"
                item = by_id.get(chunk_id)
                source = item.source if item else None
                citations.append(
                    CitationModel(
                        citation_id=citation_id,
                        claim_id=claim_id,
                        document_id=source.document_id if source else None,
                        chunk_id=chunk_id,
                        page=source.page if source else None,
                        start_char=source.start_char if source else None,
                        end_char=source.end_char if source else None,
                        resolved=item is not None,
                        source_text=item.text if item else None,
                        source_evidence_id=(
                            str(item.metadata["evidence_ids"][0])
                            if item
                            and isinstance(item.metadata.get("evidence_ids"), list)
                            and item.metadata["evidence_ids"]
                            else None
                        ),
                    )
                )
                citation_ids.append(citation_id)
            claims.append(
                ClaimModel(claim_id=claim_id, text=claim.text, citations=citation_ids)
            )
        return ObservationAnswer(
            text=payload.answer_text,
            claims=claims,
            citations=citations,
            insufficient_evidence=payload.insufficient_evidence,
            status=run_status,
            generation_status=run_status,
        )

    @staticmethod
    def _assessment(item: Assessment, index: int) -> AssessmentModel:
        """Map one runtime assessment to an explicitly identified record."""
        return AssessmentModel(
            assessment_id=f"assessment_{index}",
            assessor=item.assessor,
            dimension=item.dimension,
            label=item.label,
            score=item.score,
            evidence_ids=list(item.evidence_ids),
            metadata=item.metadata,
        )

    @staticmethod
    def _decision(
        decision: DecisionResult | None, assessments: tuple[Assessment, ...]
    ) -> DecisionModel | None:
        """Map policy action and its reason without re-evaluating policy."""
        if decision is None:
            return None
        return DecisionModel(
            action=str(decision.action),
            reason=(
                str(decision.metadata["reason"])
                if "reason" in decision.metadata
                else None
            ),
            assessment_ids=[
                f"assessment_{index}" for index, _ in enumerate(assessments, 1)
            ],
            metadata=decision.metadata,
        )

    @staticmethod
    def _legacy_trace(
        events: list[TraceEvent] | tuple[TraceEvent, ...],
    ) -> list[TraceSpan]:
        """Represent existing append-only stage events in the canonical trace shape."""
        return [
            TraceSpan(
                span_id=f"legacy_{index}",
                name=event.stage,
                started_at=event.timestamp,
                ended_at=event.timestamp,
                duration_ms=event.elapsed_seconds * 1000,
                status=event.status,
            )
            for index, event in enumerate(events, 1)
        ]

    @staticmethod
    def _datetime(value: object) -> datetime | None:
        """Parse runner metadata timestamps while treating malformed values absent."""
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                return None
        return None
