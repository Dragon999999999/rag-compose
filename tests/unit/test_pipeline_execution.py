"""Verify deterministic query execution strategies and decision policies."""

import pytest

from llm_grounding.decision.policies import (
    AdaptiveRetrievalPolicy,
    AlwaysAcceptPolicy,
    ConsensusPolicy,
    FallbackRetrievalPolicy,
    ThresholdPolicy,
    TokenBudgetPolicy,
    TopKPolicy,
)
from llm_grounding.domain.answers import (
    Answer,
    Assessment,
    DecisionAction,
    ReasoningResult,
)
from llm_grounding.domain.corpus import CorpusRequest
from llm_grounding.infrastructure.config.models import PipelineExecutionConfig
from llm_grounding.orchestration.basic_rag import BasicRagRunner


def _result() -> ReasoningResult:
    """Build a minimal candidate for policy tests."""
    return ReasoningResult("result", {})


def _assessment(
    score: float | None = 0.8,
    *,
    label: str | None = "pass",
    dimension: str = "relevance",
    evidence_ids: tuple[str, ...] = ("chunk-1",),
    metadata: dict[str, object] | None = None,
) -> Assessment:
    """Build one deterministic assessor measurement."""
    return Assessment(
        "result",
        "test_assessor",
        dimension,
        score,
        label,
        evidence_ids,
        metadata or {},
        label,
    )


async def test_deterministic_policies_produce_structured_actions() -> None:
    """Policies use existing measurements and never execute pipeline stages."""
    result = _result()
    assessments = (_assessment(),)

    assert (
        await AlwaysAcceptPolicy().decide(result, ())
    ).action == DecisionAction.ACCEPT
    assert (
        await ThresholdPolicy(0.7).decide(result, assessments)
    ).action == DecisionAction.ACCEPT
    top_k = await TopKPolicy(1).decide(result, assessments)
    assert top_k.action == DecisionAction.ACCEPT
    assert top_k.metadata["selected_evidence_ids"] == ["chunk-1"]
    assert (
        await TokenBudgetPolicy(10).decide(
            result, (_assessment(metadata={"token_count": 11}),)
        )
    ).action == DecisionAction.ABSTAIN
    assert (
        await ConsensusPolicy(0.5).decide(result, assessments)
    ).action == DecisionAction.ACCEPT


async def test_retrieval_policies_request_bounded_retries() -> None:
    """Fallback and adaptive policies return overrides for the orchestrator."""
    result = _result()
    assessments = (_assessment(0.1),)
    fallback = await FallbackRetrievalPolicy(
        threshold=0.5,
        parameter_overrides={"retrieval_top_k": 20},
    ).decide(result, assessments)
    adaptive = await AdaptiveRetrievalPolicy(
        threshold=0.5,
        parameter_overrides={"retrieval_top_k": 40},
    ).decide(result, assessments)

    assert fallback.action == DecisionAction.RETRIEVE_AGAIN
    assert adaptive.action == DecisionAction.RETRIEVE_AGAIN
    assert adaptive.metadata["parameter_overrides"] == {"retrieval_top_k": 40}


def test_execution_configuration_is_bounded_and_requires_complete_topology() -> None:
    """Branching and recursive settings reject incomplete or unbounded shapes."""
    branching = PipelineExecutionConfig.model_validate(
        {
            "strategy": "branching",
            "branches": [{"id": "dense"}, {"id": "lexical"}],
            "evidence_merge": "rrf",
        }
    )
    recursive = PipelineExecutionConfig.model_validate(
        {
            "strategy": "recursive",
            "max_rounds": 3,
            "retry_overrides": [{"retrieval_top_k": 20}],
        }
    )

    assert len(branching.branches) == 2
    assert recursive.max_rounds == 3


def test_unified_execution_validates_ordered_alternative_path_dependencies() -> None:
    """Alternatives reference concrete earlier paths and preserve branch locality."""
    execution = PipelineExecutionConfig.model_validate(
        {
            "max_attempts": 2,
            "alternatives": [
                {
                    "id": "primary",
                    "branches": [
                        {"id": "dense", "start_at": "retriever"},
                        {"id": "lexical", "start_at": "retriever"},
                    ],
                },
                {
                    "id": "retry",
                    "source": {
                        "alternative_id": "primary",
                        "path_id": "lexical",
                        "restart_stage": "assessor",
                    },
                    "branches": [{"id": "retry-path", "start_at": "assessor"}],
                },
            ],
        }
    )

    assert execution.alternatives[1].source is not None
    assert execution.alternatives[1].source.path_id == "lexical"


def test_unified_execution_rejects_forward_or_ambiguous_sources() -> None:
    """A retry cannot point forward or hide a branched source path."""
    with pytest.raises(ValueError, match="earlier alternative"):
        PipelineExecutionConfig.model_validate(
            {
                "alternatives": [
                    {"id": "primary"},
                    {
                        "id": "retry",
                        "source": {
                            "alternative_id": "later",
                            "path_id": "primary",
                            "restart_stage": "retriever",
                        },
                    },
                    {"id": "later"},
                ]
            }
        )

    with pytest.raises(ValueError, match="explicit source path"):
        PipelineExecutionConfig.model_validate(
            {
                "max_attempts": 2,
                "alternatives": [
                    {
                        "id": "primary",
                        "branches": [{"id": "dense"}, {"id": "lexical"}],
                    },
                    {
                        "id": "retry",
                        "source": {
                            "alternative_id": "primary",
                            "path_id": "primary",
                            "restart_stage": "assessor",
                        },
                    },
                ],
            }
        )


def test_legacy_alternative_selection_fields_are_ignored_during_normalization() -> None:
    """Stored alternatives keep loading without retaining local policies."""
    execution = PipelineExecutionConfig.model_validate(
        {
            "alternatives": [
                {
                    "id": "primary",
                    "decision_policy": {
                        "implementation": "always_accept",
                    },
                    "acceptance": {"strategy": "minimum_score", "threshold": 0.9},
                }
            ]
        }
    )

    alternative = execution.alternatives[0].model_dump()
    assert "decision_policy" not in alternative
    assert "acceptance" not in alternative
    assert execution.final_failure_policy == "best_result"


class _StubRunner(BasicRagRunner):
    """Replace the linear pass so orchestration control flow can be tested."""

    def __init__(self, execution: PipelineExecutionConfig) -> None:
        self._execution = execution
        self.calls: list[dict[str, object]] = []

    async def _run_once(self, request, history=(), **kwargs):  # type: ignore[no-untyped-def]
        """Return a retry once, then terminate with a deterministic answer."""
        del request, history
        values = dict(kwargs.get("runtime_config") or {})
        self.calls.append(values)
        retry = len(self.calls) == 1
        metadata = {
            "decision": (
                DecisionAction.RETRIEVE_AGAIN
                if retry
                else DecisionAction.ACCEPT
            ),
            "retrieved_chunk_ids": ["chunk-1"],
            "retrieval_results": [{"chunk_id": "chunk-1", "rank": 1}],
            "trace": [{"round": len(self.calls)}],
            "parameter_overrides": {"retrieval_top_k": 30} if retry else {},
        }
        return "result", Answer("answer"), metadata


class _AlternativeStubRunner(BasicRagRunner):
    """Return controlled decisions to exercise ordered alternative selection."""

    def __init__(
        self,
        execution: PipelineExecutionConfig,
        outcomes: list[tuple[DecisionAction, float]],
    ) -> None:
        self._execution = execution
        self._outcomes = outcomes
        self.calls = 0
        self._alternative_retrievers = {}
        self._alternative_summaries = {}
        self._alternative_rerankers = {}
        self._alternative_assessors = {}
        self._alternative_reasoners = {}
        self._alternative_renderers = {}

    async def _run_once(self, request, history=(), **kwargs):  # type: ignore[no-untyped-def]
        """Return the next configured decision and quality score."""
        del request, history, kwargs
        decision, score = self._outcomes[self.calls]
        self.calls += 1
        return (
            f"result-{self.calls}",
            Answer(f"answer-{self.calls}"),
            {
                "decision": decision,
                "assessment_scores": [score],
                "retrieved_chunk_ids": [f"chunk-{self.calls}"],
                "trace": [],
                "retrieval_stages": [],
                "stage_graph": [],
                "assessments": [],
                "usage": {},
            },
        )


async def test_recursive_runner_retries_and_persists_each_round() -> None:
    """Recursive execution stops on a policy action and retains all attempts."""
    runner = _StubRunner(
        PipelineExecutionConfig(
            strategy="recursive",
            max_rounds=3,
            retry_overrides=({"retrieval_top_k": 20},),
        )
    )

    _, _, metadata = await runner.run(CorpusRequest("project", "question"))

    assert len(runner.calls) == 2
    assert runner.calls[1]["retrieval_top_k"] == 20
    assert metadata["execution_strategy"] == "recursive"
    assert len(metadata["execution_attempts"]) == 2


async def test_alternatives_use_one_shared_policy_and_stop_on_accept() -> None:
    """A non-accepted candidate advances, while acceptance stops the sequence."""
    runner = _AlternativeStubRunner(
        PipelineExecutionConfig.model_validate(
            {
                "alternatives": [{"id": "primary"}, {"id": "retry"}],
                "max_attempts": 2,
            }
        ),
        [
            (DecisionAction.RETRIEVE_AGAIN, 0.2),
            (DecisionAction.ACCEPT, 0.8),
        ],
    )

    result_id, answer, metadata = await runner.run(
        CorpusRequest("project", "question")
    )

    assert runner.calls == 2
    assert result_id == "result-2"
    assert answer.text == "answer-2"
    assert metadata["alternative_id"] == "retry"
    assert metadata["termination_reason"] == "alternative_accepted"


async def test_alternatives_return_best_candidate_when_all_are_rejected() -> None:
    """Exhausted alternatives select the highest-quality candidate by default."""
    runner = _AlternativeStubRunner(
        PipelineExecutionConfig.model_validate(
            {
                "alternatives": [{"id": "primary"}, {"id": "retry"}],
                "max_attempts": 2,
            }
        ),
        [
            (DecisionAction.RETRIEVE_AGAIN, 0.9),
            (DecisionAction.ABSTAIN, 0.2),
        ],
    )

    result_id, answer, metadata = await runner.run(
        CorpusRequest("project", "question")
    )

    assert result_id == "result-1"
    assert answer.text == "answer-1"
    assert metadata["alternative_id"] == "primary"
    assert metadata["termination_reason"] == "max_attempts_reached"
