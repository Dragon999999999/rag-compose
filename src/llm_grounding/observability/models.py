"""Versioned, provider-neutral schemas for RAG run observations."""

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StageType(StrEnum):
    """Identify a broad execution stage in a RAG run."""

    RETRIEVAL = "retrieval"
    GENERATION = "generation"
    ASSESSMENT = "assessment"
    DECISION = "decision"
    RENDER = "render"
    INGESTION = "ingestion"
    RERANK = "rerank"
    FUSION = "fusion"
    FILTER = "filter"
    COMPRESSION = "compression"


class RetrievalStageType(StrEnum):
    """Identify retrieval outputs that evaluation can compare."""

    CANDIDATE_RETRIEVAL = "CANDIDATE_RETRIEVAL"
    FINAL_CONTEXT = "FINAL_CONTEXT"
    FUSION = "FUSION"
    RERANK = "RERANK"
    FILTER = "FILTER"
    COMPRESSION = "COMPRESSION"


class UsageOrigin(StrEnum):
    """Describe how a usage value was obtained."""

    PROVIDER_REPORTED = "provider_reported"
    TOKENIZER_CALCULATED = "tokenizer_calculated"
    ESTIMATED = "estimated"
    UNAVAILABLE = "unavailable"


class ConfidenceOrigin(StrEnum):
    """Identify a confidence source without implying unsupported calibration."""

    MODEL = "model"
    CALIBRATION_MODEL = "calibration_model"
    UNAVAILABLE = "unavailable"


class ObservationModel(BaseModel):
    """Base configuration shared by canonical observation records."""

    model_config = ConfigDict(extra="forbid")


class Confidence(ObservationModel):
    """Optional confidence with explicit interpretation and origin."""

    value: float | None = Field(default=None, ge=0, le=1)
    range: str = "[0,1]"
    interpretation: str | None = None
    origin: ConfidenceOrigin = ConfidenceOrigin.UNAVAILABLE


class GenerationInfo(ObservationModel):
    """Generation identity required to reproduce semantic-judge inputs."""

    model_id: str | None = None
    prompt_id: str | None = None
    prompt_version: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None


class Score(ObservationModel):
    """Query-specific score attached to one retrieved item."""

    value: float | None = None
    type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SourceRef(ObservationModel):
    """Persistent source identity and optional source coordinates."""

    document_id: str | None = None
    chunk_id: str | None = None
    ingestion_id: str | None = None
    page: int | None = None
    start_char: int | None = None
    end_char: int | None = None
    section: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedItem(ObservationModel):
    """One ordered retrieval result with text, score, and provenance."""

    retrieval_id: str
    rank: int = Field(ge=1)
    text: str | None = None
    source: SourceRef = Field(default_factory=SourceRef)
    score: Score = Field(default_factory=Score)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalStageSnapshot(ObservationModel):
    """Persisted snapshot of one retrieval stage and its ordered items."""

    stage_id: str
    stage_type: RetrievalStageType
    component_name: str | None = None
    parent_stage_ids: list[str] = Field(default_factory=list)
    items: list[RetrievedItem] = Field(default_factory=list)
    status: str = "complete"
    started_at: datetime | None = None
    ended_at: datetime | None = None
    duration_ms: float | None = Field(default=None, ge=0)
    parameters: dict[str, Any] = Field(default_factory=dict)
    removed_item_ids: list[str] = Field(default_factory=list)
    truncated: bool = False
    context_budget: int | None = None
    input_item_ids: list[str] = Field(default_factory=list)
    output_item_ids: list[str] = Field(default_factory=list)
    removed_items: list[RetrievedItem] = Field(default_factory=list)
    trace_span_id: str | None = None
    component_config: dict[str, Any] = Field(default_factory=dict)
    query_embedding_id: str | None = None


class StageSnapshot(ObservationModel):
    """One node in an arbitrary retrieval or generation processing DAG."""

    stage_id: str
    stage_type: StageType | str
    component_name: str
    parent_stage_ids: list[str] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    input_item_ids: list[str] = Field(default_factory=list)
    output_item_ids: list[str] = Field(default_factory=list)
    trace_span_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TraceSpan(ObservationModel):
    """Wall-clock and monotonic timing for one nested execution span."""

    span_id: str
    name: str
    stage_type: StageType | None = None
    parent_span_id: str | None = None
    started_at: datetime
    ended_at: datetime | None = None
    duration_ms: float | None = Field(default=None, ge=0)
    status: str = "running"
    input_count: int | None = None
    output_count: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    resource_usage: "ResourceUsage | None" = None


class ResourceUsage(ObservationModel):
    """Process or provider-reported resource measurements with explicit scope."""

    component_id: str
    span_id: str | None = None
    cpu_time_ms: float | None = Field(default=None, ge=0)
    cpu_utilization_percent: float | None = Field(default=None, ge=0, le=100)
    memory_peak_bytes: int | None = Field(default=None, ge=0)
    memory_average_bytes: int | None = Field(default=None, ge=0)
    gpu_id: str | None = None
    gpu_utilization_percent: float | None = Field(default=None, ge=0, le=100)
    gpu_memory_peak_bytes: int | None = Field(default=None, ge=0)
    measurement_scope: str
    measurement_source: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class LLMUsage(ObservationModel):
    """Usage reported or estimated for one model invocation."""

    model_id: str | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    requests: int | None = Field(default=None, ge=0)
    tool_calls: int | None = Field(default=None, ge=0)
    finish_reason: str | None = None
    usage_origin: UsageOrigin = UsageOrigin.UNAVAILABLE
    role: str = "target"
    metadata: dict[str, Any] = Field(default_factory=dict)


class RagUsage(ObservationModel):
    """Aggregated target-pipeline usage for one RAG execution."""

    llm_calls: int = Field(default=0, ge=0)
    embedding_calls: int = Field(default=0, ge=0)
    retrieval_calls: int = Field(default=0, ge=0)
    reranker_calls: int = Field(default=0, ge=0)
    context_tokens: int | None = Field(default=None, ge=0)
    total_input_tokens: int | None = Field(default=None, ge=0)
    total_output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    usage_origin: UsageOrigin = UsageOrigin.UNAVAILABLE
    resource_usage: list[ResourceUsage] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ObservationRequest(ObservationModel):
    """Input retained with an observation for reproducibility."""

    query: str
    history: list[dict[str, Any]] = Field(default_factory=list)
    conversation_history: list[dict[str, Any]] = Field(default_factory=list)
    included_history_turn_ids: list[str] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    filters: dict[str, Any] = Field(default_factory=dict)


class ObservationAnswer(ObservationModel):
    """Canonical answer text and source-backed claims."""

    text: str | None = None
    claims: list["ClaimModel"] = Field(default_factory=list)
    citations: list["CitationModel"] = Field(default_factory=list)
    insufficient_evidence: bool | None = None
    status: str = "completed"
    generation_status: str = "completed"
    confidence: Confidence = Field(default_factory=Confidence)


class ClaimModel(ObservationModel):
    """One canonical answer claim and the citations assigned to it."""

    claim_id: str
    text: str
    citations: list[str] = Field(default_factory=list)
    source_answer_span: dict[str, Any] | None = None
    answer_span: dict[str, Any] | None = None
    confidence: Confidence = Field(default_factory=Confidence)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CitationModel(ObservationModel):
    """Resolved or unresolved provenance for a claim citation."""

    citation_id: str
    claim_id: str
    document_id: str | None = None
    chunk_id: str | None = None
    page: int | None = None
    start_char: int | None = None
    end_char: int | None = None
    resolved: bool = False
    source_text: str | None = None
    source_evidence_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AssessmentModel(ObservationModel):
    """Canonical evaluation of a generated candidate."""

    assessment_id: str
    assessor: str
    dimension: str
    label: str | None = None
    score: float | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DecisionModel(ObservationModel):
    """Policy action taken after assessment."""

    action: str
    reason: str | None = None
    assessment_ids: list[str] = Field(default_factory=list)
    confidence: Confidence = Field(default_factory=Confidence)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EmbeddingDiagnostics(ObservationModel):
    """Query or document embedding identity and measured diagnostics."""

    embedding_id: str
    model_id: str
    model_version: str | None = None
    dimension: int
    normalization: bool | None = None
    distance_metric: str = "cosine"
    configuration_hash: str | None = None
    vector_norm: float | None = Field(default=None, ge=0)
    vector: list[float] | None = None
    source_id: str | None = None
    source_type: str | None = None
    latency_ms: float | None = Field(default=None, ge=0)
    invocation_count: int = Field(default=1, ge=0)


class ConversationState(ObservationModel):
    """State needed to reproduce a multi-turn evaluation turn."""

    prior_turn_ids: list[str] = Field(default_factory=list)
    included_history_turn_ids: list[str] = Field(default_factory=list)
    memory_state_id: str | None = None
    context_summary: str | None = None


class EvaluationCorpusReference(ObservationModel):
    """Isolated corpus identity used by robustness experiments."""

    corpus_id: str
    parent_corpus_id: str | None = None
    corpus_hash: str
    version: int = 1
    documents: list[dict[str, Any]] = Field(default_factory=list)
    chunks: list[dict[str, Any]] = Field(default_factory=list)
    perturbation_metadata: dict[str, Any] = Field(default_factory=dict)


class ArchitectureConfiguration(ObservationModel):
    """Effective architecture identity without credentials."""

    name: str
    version: int
    config_hash: str


class IngestionConfiguration(ObservationModel):
    """Effective document-ingestion settings."""

    parser: str
    chunker: str
    target_chunk_size: int | None = None
    overlap: int | None = None
    config_hash: str


class EmbeddingConfiguration(ObservationModel):
    """Effective embedding-space identity."""

    model_id: str
    version: str | None = None
    dimension: int | None = None
    normalization: bool | None = None
    embedding_hash: str | None = None


class RetrievalConfiguration(ObservationModel):
    """Effective retrieval parameters."""

    type: str
    top_k: int | None = None
    similarity_metric: str | None = None
    filters: dict[str, Any] = Field(default_factory=dict)
    context_limit: int | None = None


class GenerationConfiguration(ObservationModel):
    """Effective generation parameters with secret-free model identity."""

    model_id: str
    version: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class PromptConfiguration(ObservationModel):
    """Versioned prompt identity."""

    prompt_id: str | None = None
    prompt_version: str | None = None
    prompt_hash: str | None = None


class CorpusConfiguration(ObservationModel):
    """Corpus identity associated with the observation."""

    corpus_id: str | None = None
    version: str | None = None
    hash: str | None = None


class EffectiveConfiguration(ObservationModel):
    """All effective, reproducibility-relevant runtime configuration."""

    architecture: ArchitectureConfiguration
    ingestion: IngestionConfiguration | None = None
    embedding: EmbeddingConfiguration | None = None
    retrieval: RetrievalConfiguration | None = None
    generation: GenerationConfiguration | None = None
    prompt: PromptConfiguration | None = None
    corpus: CorpusConfiguration | None = None


class RagRunObservation(ObservationModel):
    """Canonical persisted representation shared by HTTP, CLI, and diagnostics."""

    schema_version: str = "1"
    run_id: str
    request_id: str
    target_run_id: str | None = None
    corpus_id: str
    architecture_name: str
    architecture_version: str | int
    status: str
    request: ObservationRequest
    generation: GenerationInfo = Field(default_factory=GenerationInfo)
    answer: ObservationAnswer | None = None
    retrieval: list[RetrievalStageSnapshot] = Field(default_factory=list)
    stages: list[StageSnapshot] = Field(default_factory=list)
    query_embedding_id: str | None = None
    document_embedding_ids: list[str] = Field(default_factory=list)
    embedding: EmbeddingDiagnostics | None = None
    ingestion: list[TraceSpan] = Field(default_factory=list)
    assessments: list[AssessmentModel] = Field(default_factory=list)
    decision: DecisionModel | None = None
    trace: list[TraceSpan] = Field(default_factory=list)
    usage: RagUsage = Field(default_factory=RagUsage)
    effective_configuration: EffectiveConfiguration | None = None
    configuration_hash: str | None = None
    conversation_id: str | None = None
    turn_id: str | None = None
    turn_index: int | None = Field(default=None, ge=0)
    conversation_state: ConversationState | None = None
    evaluation_corpus: EvaluationCorpusReference | None = None
    benchmark_case_id: str | None = None
    experiment_id: str | None = None
    seed: int | None = None
    perturbation_metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    completed_at: datetime | None = None
    error: str | None = None
