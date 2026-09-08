"""Minimal experiment inputs and metric outputs, outside production assessment."""

from dataclasses import dataclass

from llm_grounding.domain.runs import RagRequest, RagRun


@dataclass(frozen=True)
class TestCase:
    """Identify one reproducible application request in an experiment."""

    case_id: str
    request: RagRequest


@dataclass(frozen=True)
class TestSuite:
    """Group experiment cases under a stable human-readable identity."""

    name: str
    cases: tuple[TestCase, ...]


@dataclass(frozen=True)
class MetricResult:
    """Report a named check with explanation and completed-run identity."""

    name: str
    passed: bool
    run_id: str
    detail: str


@dataclass(frozen=True)
class CaseEvaluation:
    """Associate one executed case, its retained run, and offline metrics."""

    case_id: str
    run: RagRun
    metrics: tuple[MetricResult, ...]
