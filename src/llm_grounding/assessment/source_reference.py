"""Deterministically validate model-nominated IDs against retrieved project chunks."""

from typing import Protocol, cast

from llm_grounding.domain.answers import Assessment, JsonValue, ReasoningResult
from llm_grounding.domain.corpus import GroundingContext
from llm_grounding.domain.evidence import EvidenceUnit, RetrievalResult
from llm_grounding.reasoning.basic_grounded import GroundedResponsePayload


class SourceRecordRepository(Protocol):
    """Provide canonical evidence records for source-reference validation.

    Implementations return a mapping keyed by canonical evidence ID. The
    assessor relies on the values retaining document identity and provenance
    metadata for grounding checks.
    """

    async def evidence_by_ids(
        self, evidence_ids: tuple[str, ...]
    ) -> dict[str, EvidenceUnit]:
        """Load canonical evidence records keyed by evidence ID."""
        ...


class SourceReferenceAssessor:
    """Check cited chunk IDs against the active project's retrieved context."""

    def __init__(self, repository: SourceRecordRepository | None = None) -> None:
        """Optionally retain the canonical repository for source-record checks."""
        self._repository = repository

    async def assess_context(
        self, result: ReasoningResult, context: GroundingContext
    ) -> tuple[Assessment, ...]:
        """Return a PASS/FAIL assessment without making semantic entailment claims."""
        payload = GroundedResponsePayload.model_validate(result.payload)
        known = {item.chunk.chunk_id for item in context.chunks}
        referenced = {
            chunk_id for claim in payload.claims for chunk_id in claim.source_chunk_ids
        }
        unresolved = result.metadata.get("unresolved_source_refs", [])
        unresolved_values = unresolved if isinstance(unresolved, list) else []
        unresolved_refs = [
            value for value in unresolved_values if isinstance(value, str)
        ]
        unknown = sorted((referenced - known) | set(unresolved_refs))
        source_records = await self._source_records(context)
        missing_evidence = sorted(
            reference.evidence_id
            for item in context.chunks
            for reference in item.chunk.evidence_refs
            if source_records is not None
            and reference.evidence_id not in source_records
        )
        mismatched_evidence = sorted(
            reference.evidence_id
            for item in context.chunks
            for reference in item.chunk.evidence_refs
            if source_records is not None
            and reference.evidence_id in source_records
            and getattr(source_records[reference.evidence_id], "document_id", None)
            != item.chunk.document_id
        )
        invalid_offsets = sorted(
            reference.evidence_id
            for item in context.chunks
            for reference in item.chunk.evidence_refs
            if reference.start < 0 or reference.end < reference.start
        )
        uncited_claims = [
            index
            for index, claim in enumerate(payload.claims)
            if not claim.source_chunk_ids
        ]
        label = (
            "pass"
            if (
                not unknown
                and not uncited_claims
                and not invalid_offsets
                and not missing_evidence
                and not mismatched_evidence
            )
            else "fail"
        )
        uncited_claim_values: list[JsonValue] = list(uncited_claims)
        missing_evidence_values: list[JsonValue] = [
            str(value) for value in missing_evidence
        ]
        mismatched_evidence_values: list[JsonValue] = [
            str(value) for value in mismatched_evidence
        ]
        metadata: dict[str, JsonValue] = {
            "unknown_chunk_ids": [str(chunk_id) for chunk_id in unknown],
            "unresolved_source_refs": list(unresolved_refs),
            "uncited_claim_indexes": uncited_claim_values,
            "invalid_source_offsets": [str(value) for value in invalid_offsets],
            "missing_evidence_ids": missing_evidence_values,
            "mismatched_evidence_ids": mismatched_evidence_values,
            "project_id": context.request.project_id,
        }
        return (
            Assessment(
                result.result_id,
                "source_reference",
                "source_reference",
                1.0 if not unknown else 0.0,
                label,
                tuple(sorted(referenced & known)),
                metadata,
                (
                    "passed"
                    if not unknown
                    and not uncited_claims
                    and not invalid_offsets
                    and not missing_evidence
                    and not mismatched_evidence
                    else "failed"
                ),
            ),
        )

    async def _source_records(
        self, context: GroundingContext
    ) -> dict[str, EvidenceUnit] | None:
        """Load canonical evidence when a repository was supplied by the runtime."""
        if self._repository is None:
            return None
        evidence_ids = tuple(
            reference.evidence_id
            for item in context.chunks
            for reference in item.chunk.evidence_refs
        )
        loader = getattr(self._repository, "evidence_by_ids", None)
        if not callable(loader):
            return None
        records = await loader(evidence_ids)
        if not isinstance(records, dict):
            raise TypeError(
                "Source record retrieval must return a dictionary keyed by "
                "evidence ID"
            )
        typed_records = cast(dict[str, EvidenceUnit], records)
        if not all(
            isinstance(record, EvidenceUnit) for record in typed_records.values()
        ):
            raise TypeError(
                "Source record retrieval returned an invalid evidence record"
            )
        return typed_records

    async def assess(
        self, result: ReasoningResult, evidence: tuple[RetrievalResult, ...]
    ) -> tuple[Assessment, ...]:
        """Validate references against a temporary evidence-only context."""
        from llm_grounding.domain.corpus import (
            ChunkRetrievalResult,
            CorpusRequest,
            RetrievalUnit,
        )

        chunks = tuple(
            ChunkRetrievalResult(
                RetrievalUnit(
                    item.evidence.evidence_id,
                    item.evidence.document_id,
                    "unknown",
                    item.evidence.source_uri,
                    item.evidence.text or "",
                    item.evidence.order,
                    item.evidence.page_number,
                    item.evidence.page_number,
                    (),
                ),
                item.rank,
                item.retriever_id,
                item.score,
            )
            for item in evidence
        )
        return await self.assess_context(
            result, GroundingContext(CorpusRequest("unknown", "unknown"), chunks)
        )
