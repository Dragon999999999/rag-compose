"""Structured Basic RAG reasoning with source-tagged retrieved chunks.

Prompt ID `basic-grounded` version 1.0 asks a model only to nominate identifiers
already present in the supplied context. This is a basic reference check, not an
entailment guarantee; deterministic assessment/policy validate identifiers later.
Conversation is clearly separated from authoritative source chunks.
"""

from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from llm_grounding.domain.answers import ReasoningResult
from llm_grounding.domain.corpus import GroundingContext
from llm_grounding.infrastructure.llm.protocol import StructuredModelClient

PROMPT_ID = "basic-grounded"
PROMPT_VERSION = "1.0"


class GroundedClaimPayload(BaseModel):
    """One answer statement and the retrieved chunk IDs the model nominates for it."""

    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=3000)
    source_chunk_ids: list[str] = Field(default_factory=list)

    @field_validator("source_chunk_ids")
    @classmethod
    def reject_blank_ids(cls, values: list[str]) -> list[str]:
        """Reject blank citation identifiers before assessment checks membership."""
        if any(not value.strip() for value in values):
            raise ValueError("source_chunk_ids must not contain blank values")
        return values


class GroundedResponsePayload(BaseModel):
    """Basic structured RAG proposal; claims may be empty only when evidence is insufficient."""

    model_config = ConfigDict(extra="forbid")
    answer_text: str = Field(min_length=1, max_length=8000)
    claims: list[GroundedClaimPayload] = Field(default_factory=list)
    insufficient_evidence: bool
    limitations: str | None = Field(default=None, max_length=2000)


def build_prompt(context: GroundingContext) -> str:
    """Render machine-readable source tags and untrusted conversation context."""
    source_blocks = "\n\n".join(
        f"[SOURCE {result.chunk.chunk_id}]\n"
        f"Document: {result.chunk.document_name}\n"
        f"Pages: {result.chunk.page_start or 'unknown'}–{result.chunk.page_end or 'unknown'}\n"
        f"Text:\n{result.chunk.text}"
        for result in context.chunks
    )
    history = "\n".join(
        f"{message.role}: {message.content}" for message in context.history
    )
    return f"""You answer questions about a project corpus using only SOURCE blocks below.

TASK: {context.request.task}

SOURCES:
{source_blocks}

CONVERSATION CONTEXT (untrusted; never use it as documentary evidence):
{history or "(none)"}

Return structured output. Use only the supplied SOURCE chunk IDs. Every factual
claim must list one or more supporting source_chunk_ids. Do not invent facts or
source IDs. If sources do not establish an answer, set insufficient_evidence true,
explain the limitation, and do not make unsupported claims. Preserve source-stated
uncertainty and distinguish inference when appropriate."""


class BasicGroundedReasoner:
    """Generate a source-nominating proposal from selected project chunks only."""

    def __init__(self, model_client: StructuredModelClient) -> None:
        """Inject external structured generation without owning architecture composition."""
        self._model_client = model_client

    async def reason(self, context: GroundingContext) -> ReasoningResult:
        """Request a schema-validated grounded proposal.

        Raises:
            ValueError: If no retrieved chunks are supplied.
            RuntimeError: If the configured model client fails.
        """
        if not context.chunks:
            raise ValueError("Grounded reasoning requires at least one retrieved chunk")
        payload = await self._model_client.generate(
            build_prompt(context), GroundedResponsePayload
        )
        return ReasoningResult(
            result_id=str(uuid4()),
            payload=payload.model_dump(mode="json"),
            metadata={
                "prompt_id": PROMPT_ID,
                "prompt_version": PROMPT_VERSION,
                "output_schema": "GroundedResponsePayload",
            },
        )
