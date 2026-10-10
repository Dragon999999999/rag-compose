"""Acceptance checks for statistical experiment identity."""

from llm_grounding.application.eval_corpus_service import EvalCorpusService


def test_statistical_cases_preserve_experiment_and_case_identity() -> None:
    """The generated request identity changes when the benchmark case changes."""
    first = EvalCorpusService.deterministic_request_id(
        corpus_id="corpus", benchmark_case_id="case-a", experiment_id="exp", query="q"
    )
    second = EvalCorpusService.deterministic_request_id(
        corpus_id="corpus", benchmark_case_id="case-b", experiment_id="exp", query="q"
    )

    assert first != second
