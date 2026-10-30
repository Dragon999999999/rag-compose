"""Acceptance checks for stage graph identity and item lineage."""

from llm_grounding.orchestration.retrieval_stages import StageGraphRecorder


def test_stage_graph_has_stable_run_local_stage_ids() -> None:
    """Stage identities are unique even when stage types repeat."""
    graph = StageGraphRecorder(run_id="run")
    first = graph.add_stage("retrieval", "dense")
    second = graph.add_stage("retrieval", "sparse")

    assert first.stage_id != second.stage_id
    assert first.stage_id.startswith("run:")
