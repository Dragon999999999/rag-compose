"""Deterministically validate model-nominated IDs against retrieved project chunks."""

from llm_grounding.domain.answers import Assessment, JsonValue, ReasoningResult
from llm_grounding.domain.corpus import GroundingContext
from llm_grounding.reasoning.basic_grounded import GroundedResponsePayload


class SourceReferenceAssessor:
    """Check chunk IDs exist in selected context and belong to the active project request."""

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
        label = "pass" if not unknown else "fail"
        metadata: dict[str, JsonValue] = {
            "unknown_chunk_ids": [str(chunk_id) for chunk_id in unknown],
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
