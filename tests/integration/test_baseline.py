"""Exercise the real architecture with a substituted generation dependency offline."""

from dataclasses import replace
from pathlib import Path

import pytest
from conftest import RecordingModelClient
from pydantic_ai.models.test import TestModel

from llm_grounding.application.cli import main
from llm_grounding.application.service import RagService
from llm_grounding.domain.documents import DocumentSource
from llm_grounding.domain.runs import RagRequest
from llm_grounding.evaluation.basic import BasicPipelineEvaluator
from llm_grounding.evaluation.models import TestCase as EvaluationCase
from llm_grounding.evaluation.models import TestSuite as EvaluationSuite
from llm_grounding.evaluation.runner import EvaluationRunner
from llm_grounding.infrastructure.config.models import ArchitectureConfig
from llm_grounding.infrastructure.llm import aqueduct
from llm_grounding.orchestration.registry import ComponentRegistry
from llm_grounding.orchestration.runner import ArchitectureRunner, RunFailedError
from llm_grounding.reasoning.basic_summary import BasicSummaryPayload
from llm_grounding.reasoning.models import ReasoningRequest


async def test_full_application_produces_run_and_answer(
    pdf_path: Path, config: ArchitectureConfig, model_client: RecordingModelClient
) -> None:
    """All production stages execute in order and retain reproducibility artifacts."""
    run = await RagService(config, model_client).process_pdf(pdf_path)
    assert run.status == "completed"
    assert run.answer is not None
    assert run.answer.payload == model_client.payload.model_dump()
    assert len(run.retrieval_output) == 3
    assert run.retrieval_history == [run.retrieval_output]
    assert run.assessments == ()
    assert run.decision is not None
    assert run.decision.reasoning is run.reasoning_output
    assert run.architecture_name == "baseline_v0"
    assert run.architecture_version == 0
    assert run.completed_at >= run.started_at
    assert run.metadata["architecture_config"] == config.model_dump(mode="json")
    assert run.metadata["source_sha256"]
    assert [event.stage for event in run.trace if event.status == "completed"] == [
        "parse",
        "build_knowledge",
        "retrieve",
        "rerank",
        "reason",
        "assess:0",
        "decide",
        "render",
    ]
    assert all(metric.passed for metric in await BasicPipelineEvaluator().evaluate(run))


async def test_repeated_invocations_have_independent_runtime_state(
    pdf_path: Path, config: ArchitectureConfig, model_client: RecordingModelClient
) -> None:
    """Reusing the facade does not leak mutable trace or request state across runs."""
    service = RagService(config, model_client)
    first = await service.process_pdf(pdf_path)
    second = await service.process_pdf(pdf_path)
    assert first.run_id != second.run_id
    assert first.trace is not second.trace
    assert first.request.source.document_id == second.request.source.document_id
    assert first.retrieval_output == second.retrieval_output


class FailingReasoner:
    """Simulate an interchangeable component failing after successful retrieval."""

    async def reason(self, request: ReasoningRequest) -> None:
        """Raise an error whose sensitive body must not enter persisted traces."""
        raise RuntimeError("sensitive provider response")


async def test_failed_run_retains_partial_results_and_sanitized_trace(
    pdf_path: Path, config: ArchitectureConfig, model_client: RecordingModelClient
) -> None:
    """Component failures remain inspectable and never look like successful answers."""
    architecture = ComponentRegistry().build(config, model_client)
    architecture = replace(architecture, reasoner=FailingReasoner())
    request = RagRequest(DocumentSource("paper", pdf_path))
    with pytest.raises(RunFailedError) as caught:
        await ArchitectureRunner(architecture).run(request)
    run = caught.value.run
    assert run.status == "failed"
    assert len(run.retrieval_output) == 3
    assert run.answer is None
    assert run.trace[-1].stage == "reason"
    assert run.trace[-1].status == "failed"
    assert "sensitive" not in str(caught.value)
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert not any(
        metric.passed for metric in await BasicPipelineEvaluator().evaluate(run)
    )


async def test_evaluation_runs_terminal_cases_outside_production(
    pdf_path: Path,
    config: ArchitectureConfig,
    model_client: RecordingModelClient,
    tmp_path: Path,
) -> None:
    """Evaluate both successful and failed cases without stopping the suite."""
    runner = ArchitectureRunner(ComponentRegistry().build(config, model_client))
    suite = EvaluationSuite(
        "smoke",
        (
            EvaluationCase("good", RagRequest(DocumentSource("paper", pdf_path))),
            EvaluationCase(
                "missing",
                RagRequest(DocumentSource("missing", tmp_path / "missing.pdf")),
            ),
        ),
    )
    evaluations = await EvaluationRunner(runner, (BasicPipelineEvaluator(),)).run(suite)
    assert len(evaluations) == 2
    assert all(metric.passed for metric in evaluations[0].metrics)
    assert not any(metric.passed for metric in evaluations[1].metrics)
    assert evaluations[1].run.status == "failed"


def test_normal_cli_uses_entire_architecture_with_pydanticai_test_model(
    pdf_path: Path,
    payload: BasicSummaryPayload,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Exercise CLI → service → real adapter, substituting only PydanticAI's model."""
    monkeypatch.setenv("AQUEDUCT_API_KEY", "test-key")
    monkeypatch.setattr(
        aqueduct,
        "OpenAIChatModel",
        lambda *args, **kwargs: TestModel(custom_output_args=payload.model_dump()),
    )
    assert main([str(pdf_path)]) == 0
    output = capsys.readouterr()
    assert "Keywords:" in output.out
    assert payload.summary in output.out
    assert "Extraction successful." in output.out
    assert "test-key" not in output.out + output.err


def test_cli_reports_missing_credentials_without_traceback(
    pdf_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Missing runtime credentials cause an actionable nonzero CLI exit."""
    with pytest.raises(SystemExit) as caught:
        main([str(pdf_path)])
    assert caught.value.code == 1
    stderr = capsys.readouterr().err
    assert "AQUEDUCT_API_KEY" in stderr
    assert "Traceback" not in stderr


def test_cli_validates_source_before_configuration(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Invalid paths are usage errors before architecture execution starts."""
    with pytest.raises(SystemExit) as caught:
        main([str(tmp_path / "missing.pdf")])
    assert caught.value.code == 2
    assert "PDF file not found" in capsys.readouterr().err
