"""Acceptance checks for architecture-specific stage diagnostics."""

from llm_grounding.orchestration.retrieval_stages import StageGraphRecorder


def test_architecture_diagnostics_preserve_split_merge_lineage() -> None:
    """Parallel retrievers and fusion remain inspectable without fabricated stages."""
    graph = StageGraphRecorder(run_id="run")
    root = graph.add_stage("rewrite", "query_rewriter", output_item_ids=("q",))
    dense = graph.add_stage(
        "retrieval", "dense", parent_stage_ids=(root.stage_id,), output_item_ids=("a",)
    )
    sparse = graph.add_stage(
        "retrieval", "sparse", parent_stage_ids=(root.stage_id,), output_item_ids=("b",)
    )
    fusion = graph.add_stage(
        "fusion",
        "fusion",
        parent_stage_ids=(dense.stage_id, sparse.stage_id),
        input_item_ids=("a", "b"),
        output_item_ids=("a",),
    )

    assert fusion.parent_stage_ids == [dense.stage_id, sparse.stage_id]
    assert fusion.input_item_ids == ["a", "b"]
