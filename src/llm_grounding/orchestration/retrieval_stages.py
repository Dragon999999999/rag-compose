"""Build comparable retrieval-stage snapshots for evaluation and diagnostics."""

from datetime import datetime
from uuid import uuid4

from llm_grounding.domain.corpus import ChunkRetrievalResult
from llm_grounding.observability.models import (
    RetrievalStageSnapshot,
    RetrievalStageType,
    RetrievedItem,
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


def build_retrieval_stages(
    results: tuple[ChunkRetrievalResult, ...],
    *,
    run_id: str | None = None,
    component_name: str = "vector_retriever",
    final_context: tuple[ChunkRetrievalResult, ...] | None = None,
) -> list[RetrievalStageSnapshot]:
    """Build candidate and final-context stages with stable item identities."""
    builder = RetrievalStageBuilder(run_id=run_id)
    candidate = builder.from_results(
        RetrievalStageType.CANDIDATE_RETRIEVAL,
        results,
        component_name=component_name,
    )
    final_results = results if final_context is None else final_context
    removed = tuple(
        item.chunk.chunk_id
        for item in results
        if item.chunk.chunk_id not in {entry.chunk.chunk_id for entry in final_results}
    )
    final = builder.from_results(
        RetrievalStageType.FINAL_CONTEXT,
        final_results,
        component_name="grounding_context",
        parent_stage_ids=(candidate.stage_id,),
        removed_item_ids=removed,
    )
    return [candidate, final]
