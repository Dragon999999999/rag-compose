"""Deterministically render validated chunk citations and compact source excerpts."""

from llm_grounding.domain.answers import Answer, DecisionAction, DecisionResult
from llm_grounding.domain.corpus import GroundingContext
from llm_grounding.reasoning.basic_grounded import GroundedResponsePayload
from llm_grounding.rendering.chunk_references import chunk_reference_label


class GroundedAnswerRenderer:
    """Resolve source markers from retrieved chunks, never free-form citations."""

    async def render_context(
        self, decision: DecisionResult, context: GroundingContext
    ) -> Answer:
        """Render accepted claims or a clear insufficiency answer with source listings.

        Raises:
            ValueError: If policy rejected output or a claim references an unselected
                chunk.
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
                f"{text}\n\n{self._retrieved_chunk_listing(context, ())}",
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
            citation_labels = " ".join(
                f"[{chunk_reference_label(chunks[chunk_id])}]"
                for chunk_id in claim.source_chunk_ids
            )
            lines.append(f"{claim.text} {citation_labels}")
            referenced.extend(claim.source_chunk_ids)
        body = "\n".join(lines) if lines else payload.answer_text
        unique_ids = tuple(dict.fromkeys(referenced))
        return Answer(
            f"{body}\n\n{self._retrieved_chunk_listing(context, unique_ids)}",
            payload=payload.model_dump(mode="json"),
            evidence_ids=unique_ids,
            metadata={"insufficient_evidence": False},
        )

    @staticmethod
    def _retrieved_chunk_listing(
        context: GroundingContext, used_chunk_ids: tuple[str, ...]
    ) -> str:
        """Render every retrieved chunk, visibly distinguishing cited support."""
        used = set(used_chunk_ids)
        lines = ["Retrieved chunks"]
        for result in context.chunks:
            chunk = result.chunk
            state = "USED" if chunk.chunk_id in used else "retrieved"
            page_range = (
                f"{chunk.page_start or 'unknown'}–{chunk.page_end or 'unknown'}"
            )
            lines.extend(
                (
                    f"[{state}] {chunk_reference_label(chunk)}",
                    f"Document: {chunk.document_name}",
                    f"Pages: {page_range}",
                    f"Excerpt: {chunk.text[:500]}",
                    "",
                )
            )
        return "\n".join(lines).rstrip()
