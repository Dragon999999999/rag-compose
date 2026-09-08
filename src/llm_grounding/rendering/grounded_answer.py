"""Deterministically render validated chunk citations and compact source excerpts."""

from llm_grounding.domain.answers import Answer, DecisionAction, DecisionResult
from llm_grounding.domain.corpus import GroundingContext
from llm_grounding.reasoning.basic_grounded import GroundedResponsePayload


class GroundedAnswerRenderer:
    """Resolve source markers from actual retrieved chunks; never trust free-form citations."""

    async def render_context(
        self, decision: DecisionResult, context: GroundingContext
    ) -> Answer:
        """Render accepted claims or a clear insufficiency answer with real source listings.

        Raises:
            ValueError: If policy rejected output or a claim references an unselected chunk.
        """
        if decision.action == DecisionAction.REJECT:
            raise ValueError("Grounded renderer refuses a rejected decision")
        payload = GroundedResponsePayload.model_validate(decision.reasoning.payload)
        chunks = {result.chunk.chunk_id: result.chunk for result in context.chunks}
        if decision.action == DecisionAction.ABSTAIN:
            text = f"Insufficient evidence: {payload.answer_text}"
            if payload.limitations:
                text += f"\n\nLimitations: {payload.limitations}"
            return Answer(
                text,
                payload=payload.model_dump(mode="json"),
                metadata={"insufficient_evidence": True},
            )
        lines: list[str] = []
        referenced: list[str] = []
        for claim in payload.claims:
            if not claim.source_chunk_ids:
                raise ValueError("Accepted grounded claim has no source chunk IDs")
            if any(chunk_id not in chunks for chunk_id in claim.source_chunk_ids):
                raise ValueError(
                    "Accepted grounded claim references an unselected chunk"
                )
            lines.append(
                f"{claim.text} {' '.join(f'[{chunk_id}]' for chunk_id in claim.source_chunk_ids)}"
            )
            referenced.extend(claim.source_chunk_ids)
        body = "\n".join(lines) if lines else payload.answer_text
        unique_ids = tuple(dict.fromkeys(referenced))
        source_lines: list[str] = []
        for chunk_id in unique_ids:
            chunk = chunks[chunk_id]
            source_lines.append(
                f"[{chunk_id}]\nDocument: {chunk.document_name}\n"
                f"Pages: {chunk.page_start or 'unknown'}–{chunk.page_end or 'unknown'}\n"
                f"Excerpt: {chunk.text[:500]}"
            )
        sources = "\n\n".join(source_lines)
        return Answer(
            f"{body}\n\nSources\n\n{sources}",
            payload=payload.model_dump(mode="json"),
            evidence_ids=unique_ids,
            metadata={"insufficient_evidence": False},
        )
