"""Capability terms shared by assessor and decision-policy configuration.

Terms are deliberately small, stable identifiers rather than Python class names.
Assessors advertise measurements they produce and decision policies advertise
the measurements they require. The pipeline registry uses these terms to reject
compositions whose decision stage cannot consume the configured assessments.
"""

from collections.abc import Iterable


def assessor_default_output_terms(implementation: str) -> frozenset[str]:
    """Return the declared measurement terms for an assessor implementation."""
    terms = {
        "noop": (),
        "source_reference": ("assessment:source_reference", "assessment:label"),
        "retrieval_confidence": (
            "assessment:retrieval_confidence",
            "assessment:numeric_score",
            "assessment:label",
        ),
        "relevance": (
            "assessment:relevance",
            "assessment:numeric_score",
            "assessment:label",
        ),
        "evidence_sufficiency": (
            "assessment:evidence_sufficiency",
            "assessment:numeric_score",
            "assessment:label",
        ),
        "diversity": (
            "assessment:diversity",
            "assessment:numeric_score",
            "assessment:label",
        ),
    }
    return frozenset(terms.get(implementation, ()))


def decision_default_input_terms(
    implementation: str,
    assessment_dimension: str | None = None,
) -> frozenset[str]:
    """Return the measurement terms required by a decision policy."""
    if implementation == "basic_grounding":
        return frozenset({"assessment:source_reference"})
    if implementation in {
        "threshold",
        "top_k",
        "fallback_retrieval",
        "adaptive_retrieval",
    }:
        if assessment_dimension:
            return frozenset({f"assessment:{assessment_dimension}"})
        return frozenset({"assessment:numeric_score"})
    if implementation == "consensus":
        return frozenset({"assessment:label"})
    if implementation == "token_budget":
        return frozenset({"context:token_count"})
    return frozenset()


def decision_default_output_terms(implementation: str) -> frozenset[str]:
    """Return the decision actions advertised by a policy implementation."""
    outputs = {
        "basic_grounding": ("decision:accept", "decision:abstain", "decision:reject"),
        "always_accept": ("decision:accept",),
        "threshold": ("decision:accept", "decision:abstain"),
        "top_k": ("decision:accept",),
        "token_budget": ("decision:accept", "decision:abstain"),
        "consensus": ("decision:accept", "decision:abstain"),
        "fallback_retrieval": ("decision:accept", "decision:retrieve_again"),
        "adaptive_retrieval": ("decision:accept", "decision:retrieve_again"),
        "select_branches": ("decision:select_branches",),
        "merge_branches": ("decision:merge_branches",),
    }
    return frozenset(outputs.get(implementation, ()))


def missing_contract_terms(
    required: Iterable[str], available: Iterable[str]
) -> frozenset[str]:
    """Return required terms not advertised by the paired components."""
    return frozenset(set(required) - set(available))
