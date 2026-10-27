"""Shared candidate output, assessment, policy, and final answer contracts.

Payloads are JSON-compatible architecture-specific data. Their schema is owned
by the producing reasoner and checked by the corresponding renderer. Assessments
measure properties; decisions authorize output; answers contain presentation.
These records perform no I/O and carry no provider or framework objects.
"""

from dataclasses import dataclass, field
from enum import StrEnum

from llm_grounding.domain.claims import Claim

type JsonValue = (
    str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]
)


@dataclass(frozen=True)
class ReasoningResult:
    """Candidate intellectual output, retained before any acceptance policy."""

    result_id: str
    payload: dict[str, JsonValue]
    claims: tuple[Claim, ...] = ()
    metadata: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True)
class Assessment:
    """Measure a target in one dimension; absent scores mean no numeric measure."""

    target_id: str
    assessor: str
    dimension: str
    score: float | None = None
    label: str | None = None
    evidence_ids: tuple[str, ...] = ()
    metadata: dict[str, JsonValue] = field(default_factory=dict)
    status: str | None = None


class DecisionAction(StrEnum):
    """Describe authorization or bounded control flow for a query run."""

    ACCEPT = "accept"
    REJECT = "reject"
    ABSTAIN = "abstain"
    RETRIEVE_AGAIN = "retrieve_again"
    SELECT_BRANCHES = "select_branches"
    MERGE_BRANCHES = "merge_branches"


@dataclass(frozen=True)
class DecisionResult:
    """Record policy action and the candidate to which that action applies."""

    action: DecisionAction
    reasoning: ReasoningResult
    metadata: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True)
class Answer:
    """Expose rendered text and structured data independently of terminal output."""

    text: str
    payload: dict[str, JsonValue] = field(default_factory=dict)
    claims: tuple[Claim, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    metadata: dict[str, JsonValue] = field(default_factory=dict)
