"""Tests for arbitrary stage-DAG recording and item lineage."""

import pytest

from llm_grounding.orchestration.retrieval_stages import StageGraphRecorder


def test_stage_graph_supports_parallel_branches_and_merge() -> None:
    """A graph retains same-type branches and their merge lineage."""
    graph = StageGraphRecorder(run_id="run")
    root = graph.add_stage("rewrite", "query_rewriter", output_item_ids=("q1",))
    left = graph.add_stage(
        "retrieval", "dense", parent_stage_ids=(root.stage_id,), output_item_ids=("a",)
    )
    right = graph.add_stage(
        "retrieval", "sparse", parent_stage_ids=(root.stage_id,), output_item_ids=("b",)
    )
    merged = graph.add_stage(
        "fusion",
        "weighted_fusion",
        parent_stage_ids=(left.stage_id, right.stage_id),
        input_item_ids=("a", "b"),
        output_item_ids=("a",),
    )

    assert [stage.stage_type for stage in graph.build()] == [
        "rewrite",
        "retrieval",
        "retrieval",
        "fusion",
    ]
    assert merged.parent_stage_ids == [left.stage_id, right.stage_id]
    assert merged.input_item_ids == ["a", "b"]


def test_stage_graph_rejects_unknown_parent() -> None:
    """Lineage cannot point at a stage absent from the same run."""
    with pytest.raises(ValueError, match="Unknown parent"):
        StageGraphRecorder(run_id="run").add_stage(
            "custom", "component", parent_stage_ids=("missing",)
        )
