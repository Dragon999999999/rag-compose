"""Canonical observations and instrumentation for RAG executions.

The observability package is the boundary between runtime orchestration and
external evaluation.  Its models are provider-neutral and intentionally retain
``None`` for information that a runtime cannot measure.
"""

from llm_grounding.observability.models import (
    AssessmentModel,
    CitationModel,
    ClaimModel,
    DecisionModel,
    EffectiveConfiguration,
    RagRunObservation,
    RagUsage,
    RetrievalStageSnapshot,
    RetrievedItem,
    TraceSpan,
)

__all__ = [
    "AssessmentModel",
    "ClaimModel",
    "CitationModel",
    "DecisionModel",
    "EffectiveConfiguration",
    "RagRunObservation",
    "RagUsage",
    "RetrievalStageSnapshot",
    "RetrievedItem",
    "TraceSpan",
]
