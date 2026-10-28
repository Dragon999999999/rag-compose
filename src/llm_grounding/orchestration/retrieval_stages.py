"""Build comparable retrieval-stage snapshots for evaluation and diagnostics."""

from datetime import datetime
from uuid import uuid4

from llm_grounding.domain.corpus import ChunkRetrievalResult
from llm_grounding.observability.models import (
    RetrievalStageSnapshot,
    RetrievalStageType,
    RetrievedItem,
    StageSnapshot,
)
from llm_grounding.observability.retrieved_items import retrieved_items


class RetrievalStageBuilder:
    """Create stage records with stable IDs and explicit empty/unavailable state."""

    def __init__(self, *, run_id: str | None = None) -> None:
        """Use a run-local namespace so repeated stage types remain distinct."""
        self._run_id = run_id or str(uuid4())
        self._counter = 0

    def build(
        self,
        stage_type: RetrievalStageType,
        items: list[RetrievedItem] | tuple[RetrievedItem, ...],
        *,
        component_name: str | None = None,
        parent_stage_ids: tuple[str, ...] = (),
        started_at: datetime | None = None,
        ended_at: datetime | None = None,
        duration_ms: float | None = None,
        parameters: dict[str, object] | None = None,
        removed_item_ids: tuple[str, ...] = (),
        truncated: bool = False,
        context_budget: int | None = None,
        status: str | None = None,
    ) -> RetrievalStageSnapshot:
        """Build one snapshot and distinguish empty from unavailable output."""
        self._counter += 1
        stage_id = f"{self._run_id}:{stage_type.value.lower()}:{self._counter}"
        return RetrievalStageSnapshot(
            stage_id=stage_id,
            stage_type=stage_type,
            component_name=component_name,
            parent_stage_ids=list(parent_stage_ids),
            items=list(items),
            status=status or ("empty" if not items else "complete"),
            started_at=started_at,
            ended_at=ended_at,
            duration_ms=duration_ms,
            parameters=parameters or {},
            removed_item_ids=list(removed_item_ids),
            truncated=truncated,
            context_budget=context_budget,
        )

    def from_results(
        self,
        stage_type: RetrievalStageType,
        results: tuple[ChunkRetrievalResult, ...],
        **kwargs: object,
    ) -> RetrievalStageSnapshot:
        """Map runtime retrieval results and build the corresponding stage."""
        return self.build(stage_type, retrieved_items(results), **kwargs)


class StageGraphRecorder:
    """Record an arbitrary run-local DAG with explicit item lineage."""

    def __init__(self, *, run_id: str | None = None) -> None:
        """Use stable run-local identifiers while allowing repeated stage types."""
        self._run_id = run_id or str(uuid4())
        self._counter = 0
        self._stages: list[StageSnapshot] = []

    @property
    def stages(self) -> tuple[StageSnapshot, ...]:
        """Return stages in insertion order, independent of completion order."""
        return tuple(self._stages)

    def add_stage(
        self,
        stage_type: str,
        component_name: str,
        *,
        parent_stage_ids: tuple[str, ...] = (),
        input_item_ids: tuple[str, ...] = (),
        output_item_ids: tuple[str, ...] = (),
        parameters: dict[str, object] | None = None,
        trace_span_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> StageSnapshot:
        """Append one stage node, supporting parallel branches and custom types."""
        known_ids = {stage.stage_id for stage in self._stages}
        unknown = set(parent_stage_ids) - known_ids
        if unknown:
            raise ValueError(f"Unknown parent stage(s): {sorted(unknown)}")
        self._counter += 1
        stage = StageSnapshot(
            stage_id=f"{self._run_id}:stage:{self._counter}",
            stage_type=stage_type,
            component_name=component_name,
            parent_stage_ids=list(parent_stage_ids),
            parameters=parameters or {},
            input_item_ids=list(input_item_ids),
            output_item_ids=list(output_item_ids),
            trace_span_id=trace_span_id,
            metadata=metadata or {},
        )
        self._stages.append(stage)
        return stage

    def build(self) -> list[StageSnapshot]:
        """Return a serializable copy of the recorded DAG."""
        return list(self._stages)


def build_retrieval_stages(
    results: tuple[ChunkRetrievalResult, ...],
    *,
    run_id: str | None = None,
    component_name: str = "vector_retriever",
    reranked_context: tuple[ChunkRetrievalResult, ...] | None = None,
    final_context: tuple[ChunkRetrievalResult, ...] | None = None,
) -> list[RetrievalStageSnapshot]:
    """Build candidate, reranking, and final-context stages with stable IDs.

    ``results`` is the candidate retrieval output. When ``reranked_context`` is
    supplied it represents the output of the reranker before context selection;
    ``final_context`` then represents the exact evidence passed to generation.
    """
    builder = RetrievalStageBuilder(run_id=run_id)
    candidate = builder.from_results(
        RetrievalStageType.CANDIDATE_RETRIEVAL,
        results,
        component_name=component_name,
    )
    stages = [candidate]
    prior_stage = candidate
    prior_results = results
    if reranked_context is not None:
        rerank_removed = tuple(
            item.chunk.chunk_id
            for item in results
            if item.chunk.chunk_id
            not in {entry.chunk.chunk_id for entry in reranked_context}
        )
        rerank = builder.from_results(
            RetrievalStageType.RERANK,
            reranked_context,
            component_name="reranker",
            parent_stage_ids=(candidate.stage_id,),
            removed_item_ids=rerank_removed,
        )
        rerank.input_item_ids = [item.retrieval_id for item in candidate.items]
        rerank.output_item_ids = [item.retrieval_id for item in rerank.items]
        rerank.removed_items = [
            item for item in candidate.items if item.retrieval_id in rerank_removed
        ]
        stages.append(rerank)
        prior_stage = rerank
        prior_results = reranked_context

    final_results = prior_results if final_context is None else final_context
    removed = tuple(
        item.chunk.chunk_id
        for item in prior_results
        if item.chunk.chunk_id not in {entry.chunk.chunk_id for entry in final_results}
    )
    final = builder.from_results(
        RetrievalStageType.FINAL_CONTEXT,
        final_results,
        component_name="grounding_context",
        parent_stage_ids=(prior_stage.stage_id,),
        removed_item_ids=removed,
    )
    candidate.input_item_ids = []
    candidate.output_item_ids = [item.retrieval_id for item in candidate.items]
    final.input_item_ids = [item.retrieval_id for item in prior_stage.items]
    final.output_item_ids = [item.retrieval_id for item in final.items]
    final.removed_items = [
        item for item in prior_stage.items if item.retrieval_id in removed
    ]
    stages.append(final)
    return stages


def build_retrieval_stage_graph(
    results: tuple[ChunkRetrievalResult, ...],
    *,
    run_id: str | None = None,
    reranked_context: tuple[ChunkRetrievalResult, ...] | None = None,
    final_context: tuple[ChunkRetrievalResult, ...] | None = None,
) -> list[StageSnapshot]:
    """Build generic retrieval lineage for the basic architecture."""
    snapshots = build_retrieval_stages(
        results,
        run_id=run_id,
        reranked_context=reranked_context,
        final_context=final_context,
    )
    graph = StageGraphRecorder(run_id=run_id)
    stage_ids: dict[str, str] = {}
    for snapshot in snapshots:
        graph_stage = graph.add_stage(
            snapshot.stage_type.value,
            snapshot.component_name or snapshot.stage_type.value.lower(),
            parent_stage_ids=tuple(
                stage_ids[parent_id] for parent_id in snapshot.parent_stage_ids
            ),
            input_item_ids=tuple(snapshot.input_item_ids),
            output_item_ids=tuple(snapshot.output_item_ids),
            metadata={
                "status": snapshot.status,
                "removed_item_ids": snapshot.removed_item_ids,
            },
        )
        stage_ids[snapshot.stage_id] = graph_stage.stage_id
    return graph.build()
