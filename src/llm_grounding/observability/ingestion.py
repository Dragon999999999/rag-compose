"""Canonical ingestion inventory and trace records for evaluation."""

import json
from hashlib import sha256
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from llm_grounding.domain.corpus import Document, DocumentIngestion, RetrievalUnit
from llm_grounding.domain.evidence import EvidenceUnit
from llm_grounding.ingestion.models import IngestionConfig
from llm_grounding.observability.models import TraceSpan


class IngestionEvidenceRecord(BaseModel):
    """Persisted evidence unit exposed to diagnostic clients."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    text: str | None = None
    page: int | None = None
    source_span: dict[str, int] | None = None
    bounding_box: tuple[float, float, float, float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestionChunkRecord(BaseModel):
    """Persisted chunk inventory entry with source mappings."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    text: str
    character_count: int = Field(ge=0)
    token_count: int | None = Field(default=None, ge=0)
    ordering: int
    source_evidence_ids: list[str] = Field(default_factory=list)
    source_spans: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestionObservation(BaseModel):
    """Complete, replayable inventory for one document ingestion."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1"
    ingestion_id: str
    document_id: str
    document_hash: str
    parser_id: str
    parser_version: str
    chunker_id: str
    chunker_version: str
    chunk_size: int
    chunk_overlap: int
    chunking_parameters: dict[str, Any] = Field(default_factory=dict)
    extracted_units: list[IngestionEvidenceRecord] = Field(default_factory=list)
    chunks: list[IngestionChunkRecord] = Field(default_factory=list)
    status: str
    chunk_count: int = Field(ge=0)
    trace: list[TraceSpan] = Field(default_factory=list)
    latency_ms: float | None = Field(default=None, ge=0)
    error: str | None = None


def build_ingestion_observation(
    document: Document,
    ingestion: DocumentIngestion,
    config: IngestionConfig,
    evidence: tuple[EvidenceUnit, ...],
    chunks: tuple[RetrievalUnit, ...],
    *,
    status: str,
    trace: list[TraceSpan] | tuple[TraceSpan, ...] = (),
    latency_ms: float | None = None,
    error: str | None = None,
) -> IngestionObservation:
    """Build inventory records from canonical evidence and chunk domain objects."""
    extracted = [
        IngestionEvidenceRecord(
            evidence_id=item.evidence_id,
            text=item.text,
            page=item.page_number,
            bounding_box=item.bounding_box,
            metadata=dict(item.metadata),
        )
        for item in evidence
    ]
    chunk_records = [
        IngestionChunkRecord(
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            character_count=len(chunk.text),
            ordering=chunk.order,
            source_evidence_ids=[ref.evidence_id for ref in chunk.evidence_refs],
            source_spans=[
                {
                    "evidence_id": ref.evidence_id,
                    "start": ref.start,
                    "end": ref.end,
                }
                for ref in chunk.evidence_refs
            ],
            metadata=dict(chunk.metadata),
        )
        for chunk in chunks
    ]
    return IngestionObservation(
        ingestion_id=ingestion.id,
        document_id=document.id,
        document_hash=document.sha256,
        parser_id=config.parser,
        parser_version=config.parser_version,
        chunker_id=config.chunker.implementation,
        chunker_version=config.chunker.version,
        chunk_size=config.chunker.target_chars,
        chunk_overlap=config.chunker.overlap_chars,
        chunking_parameters=config.chunker.model_dump(mode="json"),
        extracted_units=extracted,
        chunks=chunk_records,
        status=status,
        chunk_count=len(chunk_records),
        trace=list(trace),
        latency_ms=latency_ms,
        error=error,
    )


def observation_hash(observation: IngestionObservation) -> str:
    """Return a deterministic inventory hash useful for corpus diagnostics."""
    encoded = json.dumps(
        observation.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    ).encode()
    return sha256(encoded).hexdigest()
