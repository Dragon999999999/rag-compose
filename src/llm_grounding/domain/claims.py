"""Minimal claims for future grounding; Baseline 0 does not invent claims."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Claim:
    """Identify a proposed statement and its cited evidence, without support claims."""

    claim_id: str
    text: str
    evidence_ids: tuple[str, ...] = ()
