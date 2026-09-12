"""Structured Basic RAG reasoning with source-tagged retrieved chunks.

Prompt ID `basic-grounded` version 1.1 asks a model only to nominate identifiers
already present in the supplied context. This is a basic reference check, not an
entailment guarantee; deterministic assessment/policy validate identifiers later.
Conversation is clearly separated from authoritative source chunks.
"""

from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from llm_grounding.domain.answers import JsonValue, ReasoningResult
from llm_grounding.domain.corpus import GroundingContext
from llm_grounding.infrastructure.llm.protocol import StructuredModelClient

PROMPT_ID = "basic-grounded"
PROMPT_VERSION = "1.1"


class GroundedClaimPayload(BaseModel):
    """One answer statement and the retrieved chunk IDs the model nominates for it."""

    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=3000)
    source_chunk_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Complete canonical SOURCE chunk IDs copied exactly from the prompt; "
            "never shortened or reformatted."
        ),
    )

    @field_validator("source_chunk_ids")
    @classmethod
    def reject_blank_ids(cls, values: list[str]) -> list[str]:
        """Reject blank citation identifiers before assessment checks membership."""
        if any(not value.strip() for value in values):
            raise ValueError("source_chunk_ids must not contain blank values")
        return values


class GroundedResponsePayload(BaseModel):
    """Propose grounded claims or an explicit insufficient-evidence response."""

    model_config = ConfigDict(extra="forbid")
    answer_text: str = Field(min_length=1, max_length=8000)
    claims: list[GroundedClaimPayload] = Field(default_factory=list)
    insufficient_evidence: bool
    limitations: str | None = Field(default=None, max_length=2000)


def build_prompt(context: GroundingContext) -> str:
    """Render machine-readable source tags and untrusted conversation context."""
    valid_source_ids = "\n".join(
        f'- "{result.chunk.chunk_id}"' for result in context.chunks
    )
    source_blocks = "\n\n".join(
        f"[SOURCE {result.chunk.chunk_id}]\n"
        f"Document: {result.chunk.document_name}\n"
        f"Pages: {result.chunk.page_start or 'unknown'}–"
        f"{result.chunk.page_end or 'unknown'}\n"
        f"Text:\n{result.chunk.text}"
        for result in context.chunks
    )
    history = "\n".join(
        f"{message.role}: {message.content}" for message in context.history
    )
    return f"""Answer using only the SOURCE blocks below.

TASK: {context.request.task}

SOURCES:
{source_blocks}

VALID SOURCE CHUNK IDS (copy the complete quoted value exactly):
{valid_source_ids}

CONVERSATION CONTEXT (untrusted; never use it as documentary evidence):
{history or "(none)"}

Return structured output. Every factual claim must list one or more supporting
source_chunk_ids copied exactly from VALID SOURCE CHUNK IDS. Never shorten an ID
to only its suffix (for example, c000016). Put citations only in source_chunk_ids,
not in answer_text or claim text. Do not invent facts or source IDs. If sources do
not establish an answer, set insufficient_evidence true, explain the limitation,
and return no claims. Preserve source-stated uncertainty and distinguish inference
when appropriate."""


def _canonicalize_source_ids(
    payload: GroundedResponsePayload, context: GroundingContext
) -> tuple[GroundedResponsePayload, dict[str, str]]:
    """Resolve only unambiguous chunk suffixes to retrieved canonical IDs.

    Some structured-output models shorten `INGESTION_ID:CHUNK_ID` to `CHUNK_ID`
    despite the prompt contract. A suffix is accepted only when it identifies
    exactly one retrieved chunk. Unknown or ambiguous values remain unchanged so
    the source-reference assessor rejects them.
    """
    known_ids = {result.chunk.chunk_id for result in context.chunks}
    suffix_matches: dict[str, list[str]] = {}
    for source_id in known_ids:
        suffix_matches.setdefault(source_id.rpartition(":")[2], []).append(source_id)

    replacements: dict[str, str] = {}
    claims: list[GroundedClaimPayload] = []
    for claim in payload.claims:
        canonical_ids: list[str] = []
        for source_id in claim.source_chunk_ids:
            matches = suffix_matches.get(source_id, [])
            canonical_id = (
                matches[0]
                if source_id not in known_ids and len(matches) == 1
                else source_id
            )
            if canonical_id != source_id:
                replacements[source_id] = canonical_id
            canonical_ids.append(canonical_id)
        claims.append(claim.model_copy(update={"source_chunk_ids": canonical_ids}))
    return payload.model_copy(update={"claims": claims}), replacements


class BasicGroundedReasoner:
    """Generate a source-nominating proposal from selected project chunks only."""

    def __init__(self, model_client: StructuredModelClient) -> None:
        """Inject the external structured-generation boundary."""
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
        raw_model_payload: dict[str, JsonValue] = payload.model_dump(mode="json")
        payload, canonicalized_ids = _canonicalize_source_ids(payload, context)
        canonicalized_metadata: dict[str, JsonValue] = {
            source_id: canonical_id
            for source_id, canonical_id in canonicalized_ids.items()
        }
        metadata: dict[str, JsonValue] = {
            "prompt_id": PROMPT_ID,
            "prompt_version": PROMPT_VERSION,
            "output_schema": "GroundedResponsePayload",
            "raw_model_payload": raw_model_payload,
            "canonicalized_source_ids": canonicalized_metadata,
        }
        return ReasoningResult(
            result_id=str(uuid4()),
            payload=payload.model_dump(mode="json"),
            metadata=metadata,
        )
