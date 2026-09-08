"""Deliberate whole-document baseline selection, without semantic retrieval."""

from llm_grounding.domain.evidence import EvidenceType, RetrievalResult, source_order
from llm_grounding.retrieval.models import RetrievalRequest, RetrievalResponse


class FullDocumentRetriever:
    """Return all nonempty TEXT evidence in deterministic source order."""

    async def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        """Select all text and assign ranks without relevance scores."""
        evidence = sorted(
            (
                item
                for item in request.knowledge.evidence
                if item.evidence_type == EvidenceType.TEXT
                and item.text is not None
                and item.text.strip()
            ),
            key=source_order,
        )
        return RetrievalResponse(
            query=request.query,
            results=tuple(
                RetrievalResult(item, rank, "full_document")
                for rank, item in enumerate(evidence, start=1)
            ),
        )
