"""Opt-in structured diagnostics for Basic RAG grounding decisions.

Diagnostics are disabled by default because verbose records include retrieved
document text, conversation context, and untrusted model output. The CLI can
enable summary or verbose records for investigation and operational monitoring.
"""

import json
import logging
from enum import StrEnum

from llm_grounding.domain.answers import Assessment, DecisionResult, ReasoningResult
from llm_grounding.domain.corpus import GroundingContext

_LOGGER = logging.getLogger("llm_grounding.grounding_diagnostics")


class GroundingDiagnosticsLevel(StrEnum):
    """Control the amount of grounding-decision data emitted to logs."""

    OFF = "off"
    SUMMARY = "summary"
    VERBOSE = "verbose"


def configure_console_grounding_diagnostics() -> None:
    """Send grounding diagnostics to stderr without changing the root logger."""
    if not _LOGGER.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        _LOGGER.addHandler(handler)
    _LOGGER.setLevel(logging.INFO)
    _LOGGER.propagate = False


class GroundingDiagnostics:
    """Emit structured retrieval, generation, assessment, and decision records."""

    def __init__(
        self, level: GroundingDiagnosticsLevel = GroundingDiagnosticsLevel.OFF
    ) -> None:
        """Store the requested diagnostic detail level."""
        self._level = level

    def report_empty_retrieval(self, context: GroundingContext) -> None:
        """Record a query that had no candidate chunks to assess."""
        if self._level == GroundingDiagnosticsLevel.OFF:
            return
        self._emit(
            {
                "event": "grounding_diagnostics",
                "level": self._level,
                "request": self._request_record(context),
                "retrieval": {"chunk_count": 0, "chunks": []},
                "outcome": "no_matching_chunks",
            }
        )

    def report(
        self,
        context: GroundingContext,
        reasoning: ReasoningResult,
        assessments: tuple[Assessment, ...],
        decision: DecisionResult,
    ) -> None:
        """Record all information used to authorize or reject a response."""
        if self._level == GroundingDiagnosticsLevel.OFF:
            return
        record: dict[str, object] = {
            "event": "grounding_diagnostics",
            "level": self._level,
            "request": self._request_record(context),
            "retrieval": {
                "chunk_count": len(context.chunks),
                "chunks": [
                    {
                        "id": result.chunk.chunk_id,
                        "rank": result.rank,
                        "score": result.score,
                        "retriever_id": result.retriever_id,
                    }
                    for result in context.chunks
                ],
            },
            "assessments": [
                {
                    "assessor": assessment.assessor,
                    "dimension": assessment.dimension,
                    "label": assessment.label,
                    "score": assessment.score,
                    "evidence_ids": assessment.evidence_ids,
                    "metadata": assessment.metadata,
                }
                for assessment in assessments
            ],
            "decision": {
                "action": decision.action,
                "metadata": decision.metadata,
            },
        }
        if self._level == GroundingDiagnosticsLevel.VERBOSE:
            record["context"] = {
                "history": [
                    {"role": message.role, "content": message.content}
                    for message in context.history
                ],
                "chunks": [
                    {
                        "id": result.chunk.chunk_id,
                        "document_id": result.chunk.document_id,
                        "document_name": result.chunk.document_name,
                        "page_start": result.chunk.page_start,
                        "page_end": result.chunk.page_end,
                        "text": result.chunk.text,
                        "evidence_refs": [
                            {
                                "evidence_id": reference.evidence_id,
                                "start": reference.start,
                                "end": reference.end,
                            }
                            for reference in result.chunk.evidence_refs
                        ],
                        "metadata": result.chunk.metadata,
                    }
                    for result in context.chunks
                ],
            }
            record["llm"] = {
                "raw_response": reasoning.metadata.get(
                    "raw_model_payload", reasoning.payload
                ),
                "canonicalized_response": reasoning.payload,
                "reasoning_metadata": reasoning.metadata,
            }
        self._emit(record)

    @staticmethod
    def _request_record(context: GroundingContext) -> dict[str, object]:
        """Serialize the request independently from untrusted model output."""
        return {
            "project_id": context.request.project_id,
            "task": context.request.task,
            "task_type": context.request.task_type,
            "document_ids": context.request.document_ids,
        }

    @staticmethod
    def _emit(record: dict[str, object]) -> None:
        """Log a deterministic JSON record suitable for capture or ingestion."""
        _LOGGER.info(
            "Grounding diagnostics:\n%s", json.dumps(record, indent=2, default=str)
        )
