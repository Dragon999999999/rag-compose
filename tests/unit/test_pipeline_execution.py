"""Verify deterministic query execution strategies and decision policies."""

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
