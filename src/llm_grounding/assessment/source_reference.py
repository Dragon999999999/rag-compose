"""Deterministically validate model-nominated IDs against retrieved project chunks."""

from llm_grounding.domain.answers import Assessment, JsonValue, ReasoningResult
from llm_grounding.domain.corpus import GroundingContext
from llm_grounding.reasoning.basic_grounded import GroundedResponsePayload


class SourceReferenceAssessor:
    """Check cited chunk IDs against the active project's retrieved context."""

    async def assess_context(
        self, result: ReasoningResult, context: GroundingContext
    ) -> tuple[Assessment, ...]:
        """Return a PASS/FAIL assessment without making semantic entailment claims."""
        payload = GroundedResponsePayload.model_validate(result.payload)
        known = {item.chunk.chunk_id for item in context.chunks}
        referenced = {
            chunk_id for claim in payload.claims for chunk_id in claim.source_chunk_ids
        }
        unknown = sorted(referenced - known)
        uncited_claims = [
            index
            for index, claim in enumerate(payload.claims)
            if not claim.source_chunk_ids
        ]
        label = "pass" if not unknown and not uncited_claims else "fail"
        uncited_claim_values: list[JsonValue] = list(uncited_claims)
        metadata: dict[str, JsonValue] = {
            "unknown_chunk_ids": [str(chunk_id) for chunk_id in unknown],
            "uncited_claim_indexes": uncited_claim_values,
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
            ),
        )
