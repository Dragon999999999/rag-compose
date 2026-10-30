"""Reasoning input envelope; shared candidate results live in domain.answers."""

from dataclasses import dataclass

from llm_grounding.domain.evidence import RetrievalResult


@dataclass(frozen=True)
class ReasoningRequest:
    """Supply the task and selected evidence without prescribing orchestration."""

    task: str
    evidence: tuple[RetrievalResult, ...]
