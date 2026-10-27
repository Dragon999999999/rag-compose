"""Deterministic decision policies for bounded pipeline control flow.

Policies consume only the reasoning result and assessor measurements. They do
not call retrieval, inference, databases, or branch executors; the orchestration
layer interprets control-flow actions and applies the returned metadata.
"""

from collections.abc import Iterable

from llm_grounding.domain.answers import (
    Assessment,
    DecisionAction,
    DecisionResult,
    ReasoningResult,
)


def _scores(
    assessments: tuple[Assessment, ...], dimension: str | None = None
) -> tuple[Assessment, ...]:
    """Return numeric measurements restricted to an optional dimension."""
    return tuple(
        item
        for item in assessments
        if item.score is not None and (dimension is None or item.dimension == dimension)
    )


def _decision(
    action: DecisionAction,
    result: ReasoningResult,
    reason: str,
    **metadata: object,
) -> DecisionResult:
    """Build a decision with a stable explanation and JSON-safe metadata."""
    return DecisionResult(action, result, {"reason": reason, **metadata})


class AlwaysAcceptPolicy:
    """Authorize the candidate without inventing an assessment."""

    async def decide(
        self, result: ReasoningResult, assessments: tuple[Assessment, ...]
    ) -> DecisionResult:
        """Accept regardless of assessor availability."""
        return _decision(DecisionAction.ACCEPT, result, "always_accept")


class ThresholdPolicy:
    """Accept when all selected numeric assessor scores meet a threshold."""

    def __init__(self, threshold: float = 0.5, dimension: str | None = None) -> None:
        """Configure the minimum score and optional assessment dimension."""
        self.threshold = threshold
        self.dimension = dimension

    async def decide(
        self, result: ReasoningResult, assessments: tuple[Assessment, ...]
    ) -> DecisionResult:
        """Return abstention for missing or below-threshold measurements."""
        selected = _scores(assessments, self.dimension)
        if not selected:
            return _decision(
                DecisionAction.ABSTAIN,
                result,
                "threshold_measurement_unavailable",
                threshold=self.threshold,
                assessment_dimension=self.dimension,
            )
        passed = all(
            item.score is not None and item.score >= self.threshold
            for item in selected
        )
        return _decision(
            DecisionAction.ACCEPT if passed else DecisionAction.ABSTAIN,
            result,
            "threshold_passed" if passed else "threshold_failed",
            threshold=self.threshold,
            assessment_dimension=self.dimension,
            scores=[item.score for item in selected],
        )


class TopKPolicy:
    """Authorize the candidate and expose the highest-scoring evidence IDs."""

    def __init__(self, top_k: int = 5, dimension: str | None = None) -> None:
        """Configure the maximum number of evidence IDs to select."""
        self.top_k = top_k
        self.dimension = dimension

    async def decide(
        self, result: ReasoningResult, assessments: tuple[Assessment, ...]
    ) -> DecisionResult:
        """Return deterministic evidence selection metadata without reordering text."""
        selected = _scores(assessments, self.dimension)
        ordered = sorted(
            selected,
            key=lambda item: (
                -(item.score or 0.0),
                item.assessor,
                item.target_id,
            ),
        )
        evidence_ids: list[str] = []
        for item in ordered:
            for evidence_id in item.evidence_ids:
                if evidence_id not in evidence_ids:
                    evidence_ids.append(evidence_id)
                if len(evidence_ids) >= self.top_k:
                    break
            if len(evidence_ids) >= self.top_k:
                break
        return _decision(
            DecisionAction.ACCEPT,
            result,
            "top_k_selected",
            top_k=self.top_k,
            selected_evidence_ids=evidence_ids,
        )


class TokenBudgetPolicy:
    """Abstain when assessor-reported token estimates exceed a hard budget."""

    def __init__(self, token_budget: int) -> None:
        """Configure the maximum reported context token count."""
        self.token_budget = token_budget

    async def decide(
        self, result: ReasoningResult, assessments: tuple[Assessment, ...]
    ) -> DecisionResult:
        """Use assessor metadata token counts without estimating hidden tokens."""
        counts = [
            value
            for item in assessments
            for value in [item.metadata.get("token_count")]
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        context_count = result.metadata.get("context_token_count")
        if not counts and isinstance(context_count, (int, float)):
            counts.append(context_count)
        total = max(counts, default=0)
        within_budget = total <= self.token_budget
        return _decision(
            DecisionAction.ACCEPT if within_budget else DecisionAction.ABSTAIN,
            result,
            "token_budget_available" if within_budget else "token_budget_exceeded",
            token_budget=self.token_budget,
            token_count=total,
        )


class ConsensusPolicy:
    """Accept when assessor labels reach a deterministic agreement ratio."""

    def __init__(self, min_agreement: float = 0.5) -> None:
        """Configure the minimum share of the majority label."""
        self.min_agreement = min_agreement

    async def decide(
        self, result: ReasoningResult, assessments: tuple[Assessment, ...]
    ) -> DecisionResult:
        """Accept only a majority ``pass``/``accept`` consensus."""
        labels = [item.label or item.status for item in assessments]
        labels = [label for label in labels if label is not None]
        if not labels:
            return _decision(DecisionAction.ABSTAIN, result, "consensus_unavailable")
        counts = {label: labels.count(label) for label in set(labels)}
        majority = max(counts, key=lambda label: (-counts[label], label))
        agreement = counts[majority] / len(labels)
        accepted = majority in {"pass", "passed", "accept", "accepted"}
        accepted = accepted and agreement >= self.min_agreement
        return _decision(
            DecisionAction.ACCEPT if accepted else DecisionAction.ABSTAIN,
            result,
            "consensus_reached" if accepted else "consensus_failed",
            majority_label=majority,
            agreement=agreement,
            min_agreement=self.min_agreement,
        )


class FallbackRetrievalPolicy:
    """Request one bounded retrieval retry when assessment evidence is weak."""

    def __init__(
        self,
        threshold: float = 0.5,
        parameter_overrides: dict[str, object] | None = None,
    ) -> None:
        """Configure the weak-evidence threshold and retry overrides."""
        self.threshold = threshold
        self.parameter_overrides = dict(parameter_overrides or {})

    async def decide(
        self, result: ReasoningResult, assessments: tuple[Assessment, ...]
    ) -> DecisionResult:
        """Request another retrieval only when measurements are unavailable or weak."""
        scores = _scores(assessments)
        weak = not scores or max(item.score or 0.0 for item in scores) < self.threshold
        if weak:
            return _decision(
                DecisionAction.RETRIEVE_AGAIN,
                result,
                "fallback_retrieval_requested",
                parameter_overrides=self.parameter_overrides,
            )
        return _decision(DecisionAction.ACCEPT, result, "fallback_not_needed")


class AdaptiveRetrievalPolicy(FallbackRetrievalPolicy):
    """Request bounded retries with progressively supplied parameter overrides."""

    async def decide(
        self, result: ReasoningResult, assessments: tuple[Assessment, ...]
    ) -> DecisionResult:
        """Use the fallback action while identifying the policy as adaptive."""
        decision = await super().decide(result, assessments)
        metadata = dict(decision.metadata)
        metadata["reason"] = (
            "adaptive_retrieval_requested"
            if decision.action == DecisionAction.RETRIEVE_AGAIN
            else "adaptive_not_needed"
        )
        return DecisionResult(decision.action, decision.reasoning, metadata)


class BranchSelectionPolicy:
    """Select configured branch IDs without executing the branches."""

    def __init__(self, branch_ids: Iterable[str] = ()) -> None:
        """Configure the deterministic branch selection order."""
        self.branch_ids = tuple(branch_ids)

    async def decide(
        self, result: ReasoningResult, assessments: tuple[Assessment, ...]
    ) -> DecisionResult:
        """Return a branch-selection action for the execution orchestrator."""
        return _decision(
            DecisionAction.SELECT_BRANCHES,
            result,
            "branches_selected",
            branch_ids=list(self.branch_ids),
        )


class BranchFusionPolicy:
    """Request deterministic branch fusion at the orchestration boundary."""

    async def decide(
        self, result: ReasoningResult, assessments: tuple[Assessment, ...]
    ) -> DecisionResult:
        """Return a merge action without touching branch evidence."""
        return _decision(DecisionAction.MERGE_BRANCHES, result, "branches_merged")
