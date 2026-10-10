"""Grounded QA reasoning using request-local source references.

The model receives prompt-library-selected `chunk_N` references rather than
canonical corpus identifiers. This module resolves those aliases back to canonical
chunk IDs before the existing deterministic source-reference assessment runs.
"""

from dataclasses import dataclass
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from llm_grounding.domain.answers import JsonValue, ReasoningResult
from llm_grounding.domain.corpus import GroundingContext
from llm_grounding.infrastructure.llm.protocol import (
    GenerationResult,
    StructuredModelClient,
)
from llm_grounding.infrastructure.prompts import PromptLibrary, RenderedPrompt


class ModelClaimPayload(BaseModel):
    """One model-facing factual claim supported by request-local references."""

    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=3000)
    source_refs: list[str] = Field(
        min_length=1,
        description=(
            "Exact request-local references of retrieved evidence chunks that "
            "directly support this claim. Values must be copied from the "
            "supplied chunk references."
        ),
    )

    @field_validator("source_refs")
    @classmethod
    def reject_blank_refs(cls, values: list[str]) -> list[str]:
        """Reject blank source references before deterministic resolution."""
        if any(not value.strip() for value in values):
            raise ValueError("source_refs must not contain blank values")
        return values


class ModelGroundedResponsePayload(BaseModel):
    """Structured LLM response using only request-local source references."""

    model_config = ConfigDict(extra="forbid")
    chat_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=80,
        description=(
            "Concise conversation name requested only for the first answer of an "
            "unnamed chat; otherwise null."
        ),
    )
    turn_title: str = Field(
        min_length=1,
        max_length=120,
        description=(
            "Concise navigation title for this prompt-and-answer turn. It names "
            "the user's request without adding factual claims or source references."
        ),
    )
    answer_text: str = Field(min_length=1, max_length=8000)
    claims: list[ModelClaimPayload] = Field(default_factory=list)
    insufficient_evidence: bool
    limitations: str | None = Field(default=None, max_length=2000)

    @field_validator("chat_name")
    @classmethod
    def normalize_chat_name(cls, value: str | None) -> str | None:
        """Normalize a generated name and reject the default placeholder."""
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("chat_name must not be blank")
        if normalized.casefold() == "new chat":
            raise ValueError("chat_name must replace the default placeholder")
        return normalized

    @field_validator("turn_title")
    @classmethod
    def reject_blank_turn_title(cls, value: str) -> str:
        """Normalize the persisted navigation title and reject blank values."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("turn_title must not be blank")
        return normalized

    @model_validator(mode="after")
    def require_claims_when_evidence_is_sufficient(
        self,
    ) -> "ModelGroundedResponsePayload":
        """Require source-backed claims unless the model explicitly abstains."""
        if not self.insufficient_evidence and not self.claims:
            raise ValueError(
                "claims must not be empty when insufficient_evidence is false"
            )
        return self


class ModelInitialGroundedResponsePayload(ModelGroundedResponsePayload):
    """Require a generated conversation name for an unnamed chat's first answer."""

    chat_name: str = Field(
        min_length=1,
        max_length=80,
        description=(
            "Concise, specific conversation name for this first chat answer. It "
            "must replace the default placeholder."
        ),
    )


class GroundedClaimPayload(BaseModel):
    """One canonical internal claim whose support uses persistent chunk IDs."""

    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=3000)
    source_chunk_ids: list[str] = Field(default_factory=list)


class GroundedResponsePayload(BaseModel):
    """Canonicalized response consumed by assessment, policy, and rendering."""

    model_config = ConfigDict(extra="forbid")
    chat_name: str | None = Field(default=None, min_length=1, max_length=80)
    turn_title: str = Field(default="Earlier answer", min_length=1, max_length=120)
    answer_text: str = Field(min_length=1, max_length=8000)
    claims: list[GroundedClaimPayload] = Field(default_factory=list)
    insufficient_evidence: bool
    limitations: str | None = Field(default=None, max_length=2000)


@dataclass(frozen=True)
class PromptChunk:
    """Minimal LLM-facing representation of one selected retrieval chunk."""

    ref: str
    document_name: str
    page_start: str
    page_end: str
    text: str


@dataclass(frozen=True)
class GroundedPromptContext:
    """Bind request-local model references to canonical selected chunk IDs."""

    chunks: tuple[PromptChunk, ...]
    alias_to_chunk_id: dict[str, str]


def build_grounded_prompt_context(context: GroundingContext) -> GroundedPromptContext:
    """Assign deterministic `chunk_N` references in final retrieval order."""
    chunks: list[PromptChunk] = []
    alias_to_chunk_id: dict[str, str] = {}
    for position, result in enumerate(context.chunks, start=1):
        ref = f"chunk_{position}"
        chunk = result.chunk
        alias_to_chunk_id[ref] = chunk.chunk_id
        chunks.append(
            PromptChunk(
                ref,
                chunk.document_name,
                str(chunk.page_start) if chunk.page_start is not None else "unknown",
                str(chunk.page_end) if chunk.page_end is not None else "unknown",
                chunk.text,
            )
        )
    return GroundedPromptContext(tuple(chunks), alias_to_chunk_id)


def _history_text(context: GroundingContext) -> str:
    """Render conversation context without treating it as evidence."""
    history = "\n".join(
        f"{message.role}: {message.content}" for message in context.history
    )
    return history or "(none)"


def _canonicalize_model_payload(
    payload: ModelGroundedResponsePayload,
    prompt_context: GroundedPromptContext,
) -> tuple[GroundedResponsePayload, list[dict[str, str | None]], list[str]]:
    """Resolve model aliases to canonical IDs without guessing unknown values."""
    claims: list[GroundedClaimPayload] = []
    resolutions: list[dict[str, str | None]] = []
    unresolved_refs: list[str] = []
    for claim in payload.claims:
        canonical_ids: list[str] = []
        for source_ref in claim.source_refs:
            canonical_id = prompt_context.alias_to_chunk_id.get(source_ref)
            resolutions.append(
                {
                    "source_ref": source_ref,
                    "canonical_chunk_id": canonical_id,
                }
            )
            if canonical_id is None:
                unresolved_refs.append(source_ref)
            else:
                canonical_ids.append(canonical_id)
        claims.append(
            GroundedClaimPayload(
                text=claim.text,
                source_chunk_ids=canonical_ids,
            )
        )
    return (
        GroundedResponsePayload(
            chat_name=payload.chat_name,
            turn_title=payload.turn_title,
            answer_text=payload.answer_text,
            claims=claims,
            insufficient_evidence=payload.insufficient_evidence,
            limitations=payload.limitations,
        ),
        resolutions,
        unresolved_refs,
    )


class BasicGroundedReasoner:
    """Generate claims against aliases and return canonical internal evidence IDs."""

    def __init__(
        self,
        model_client: StructuredModelClient,
        prompts: PromptLibrary,
        prompt_id: str,
    ) -> None:
        """Inject structured generation and a configuration-selected prompt library."""
        self._model_client = model_client
        self._prompts = prompts
        self._prompt_id = prompt_id

    async def reason(self, context: GroundingContext) -> ReasoningResult:
        """Render the selected prompt, call the model, and canonicalize source refs.

        Raises:
            ValueError: If no retrieved chunks are supplied.
            RuntimeError: If the configured model client fails.
        """
        if not context.chunks:
            raise ValueError("Grounded reasoning requires at least one retrieved chunk")
        prompt_context = build_grounded_prompt_context(context)
        prompt = self._render_prompt(context, prompt_context)
        model_payload: ModelGroundedResponsePayload
        generation_usage: dict[str, JsonValue] = {}
        if context.generate_chat_name:
            generation = await self._model_client.generate(
                prompt.text, ModelInitialGroundedResponsePayload
            )
        else:
            generation = await self._model_client.generate(
                prompt.text, ModelGroundedResponsePayload
            )
        if isinstance(generation, GenerationResult):
            model_payload = generation.output
            generation_usage = {
                "usage": generation.usage.model_dump(mode="json")
                if generation.usage
                else None,
                "model_id": generation.model_id,
            }
        else:
            model_payload = generation
        raw_model_payload: dict[str, JsonValue] = model_payload.model_dump(mode="json")
        canonical_payload, resolutions, unresolved_refs = _canonicalize_model_payload(
            model_payload, prompt_context
        )
        alias_metadata: dict[str, JsonValue] = {
            source_ref: canonical_id
            for source_ref, canonical_id in prompt_context.alias_to_chunk_id.items()
        }
        resolution_metadata: list[JsonValue] = []
        for resolution in resolutions:
            resolution_metadata.append(
                {
                    "source_ref": resolution["source_ref"],
                    "canonical_chunk_id": resolution["canonical_chunk_id"],
                }
            )
        unresolved_metadata: list[JsonValue] = list(unresolved_refs)
        metadata: dict[str, JsonValue] = {
            "prompt_id": prompt.metadata.prompt_id,
            "prompt_version": prompt.metadata.version,
            "output_schema": type(model_payload).__name__,
            "raw_model_payload": raw_model_payload,
            "source_ref_aliases": alias_metadata,
            "source_ref_resolutions": resolution_metadata,
            "unresolved_source_refs": unresolved_metadata,
            "generation": generation_usage,
        }
        return ReasoningResult(
            result_id=str(uuid4()),
            payload=canonical_payload.model_dump(mode="json"),
            metadata=metadata,
        )

    def _render_prompt(
        self,
        context: GroundingContext,
        prompt_context: GroundedPromptContext,
    ) -> RenderedPrompt:
        """Render the configured template with only model-facing source data."""
        return self._prompts.render(
            self._prompt_id,
            {
                "task": context.request.task,
                "history": _history_text(context),
                "chunks": prompt_context.chunks,
                "generate_chat_name": context.generate_chat_name,
            },
        )
