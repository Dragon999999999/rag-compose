"""Whole-document keyword/summary extraction using prompt basic-summary version 1.0.

The small local prompt consumes ordered paper text and a task, and requests
BasicSummaryPayload. Keyword count and summary length are validated; 1–3 words
per keyword and 2–3 sentences are model instructions, as in the original script.
This nondeterministic extraction does not verify claims. See docs/architecture.md.
"""

from uuid import uuid4

from pydantic import BaseModel, Field

from llm_grounding.domain.answers import ReasoningResult
from llm_grounding.domain.evidence import EvidenceType, source_order
from llm_grounding.infrastructure.llm.protocol import StructuredModelClient
from llm_grounding.reasoning.models import ReasoningRequest

PROMPT_ID = "basic-summary"
PROMPT_VERSION = "1.0"


class BasicSummaryPayload(BaseModel):
    """Structured extraction preserving the original script's validation bounds."""

    keywords: list[str] = Field(
        min_length=5,
        max_length=8,
        description="5–8 topic keywords or short phrases (1–3 words each).",
    )
    summary: str = Field(
        min_length=20,
        max_length=500,
        description="A concise 2–3 sentence summary of the paper's main contribution.",
    )


def build_prompt(paper_content: str, task: str) -> str:
    """Format the versioned extraction prompt from ordered paper text and user task."""
    return f"""You are a research analyst.
Extract key information from this research paper.

PAPER CONTENT:
{paper_content}

TASK:
1. Extract 5-8 keywords or short phrases (1-3 words each) that capture the main topics
2. Write a 2-3 sentence summary of the main contribution (20-500 characters)

User request: {task}
Use information supported by the supplied paper.
Return a JSON object with 'keywords' (list of strings) and 'summary' (string)."""


class BasicSummaryReasoner:
    """Join selected text in source order and request structured model extraction."""

    def __init__(self, model_client: StructuredModelClient) -> None:
        """Inject a model adapter; construction does not make network requests."""
        self._model_client = model_client

    async def reason(self, request: ReasoningRequest) -> ReasoningResult:
        """Extract an unverified candidate and retain prompt identity.

        Raises:
            ValueError: If no nonempty textual evidence is available.
            RuntimeError: If the model adapter fails to produce validated output.
        """
        evidence = sorted(
            (
                result.evidence
                for result in request.evidence
                if result.evidence.evidence_type == EvidenceType.TEXT
                and result.evidence.text
                and result.evidence.text.strip()
            ),
            key=source_order,
        )
        if not evidence:
            raise ValueError("Basic summary requires nonempty textual evidence.")
        paper_content = "\n".join(
            item.text for item in evidence if item.text is not None
        )
        output = await self._model_client.generate(
            build_prompt(paper_content, request.task), BasicSummaryPayload
        )
        return ReasoningResult(
            result_id=str(uuid4()),
            payload=output.model_dump(mode="json"),
            metadata={
                "prompt_id": PROMPT_ID,
                "prompt_version": PROMPT_VERSION,
                "output_schema": "BasicSummaryPayload",
            },
        )
