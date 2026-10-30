"""Build canonical diagnostics and controlled raw-vector evaluation records."""

from pydantic import BaseModel, ConfigDict, Field

from llm_grounding.embedding.models import EmbeddingVector
from llm_grounding.observability.models import EmbeddingDiagnostics


class EmbeddingRecord(BaseModel):
    """Optional evaluation artifact for one sampled document or query vector."""

    model_config = ConfigDict(extra="forbid")

    vector: list[float] | None = None
    vector_norm: float | None = Field(default=None, ge=0)
    embedding_id: str
    source_id: str | None = None
    source_type: str | None = None
    model_id: str
    model_version: str | None = None


def embedding_record(
    vector: EmbeddingVector,
    *,
    source_id: str | None = None,
    source_type: str | None = None,
    model_version: str | None = None,
    include_vector: bool = False,
) -> EmbeddingRecord:
    """Create a raw-vector record only when an evaluation explicitly samples it."""
    return EmbeddingRecord(
        vector=list(vector.values) if include_vector else None,
        vector_norm=vector.vector_norm,
        embedding_id=vector.embedding_id,
        source_id=source_id,
        source_type=source_type,
        model_id=vector.model_identity,
        model_version=model_version,
    )


def embedding_diagnostics(
    vector: EmbeddingVector,
    *,
    source_id: str | None = None,
    source_type: str | None = None,
    model_version: str | None = None,
    normalization: bool | None = None,
    configuration_hash: str | None = None,
    latency_ms: float | None = None,
    include_vector: bool = False,
) -> EmbeddingDiagnostics:
    """Map a vector to diagnostics while omitting raw values by default."""
    return EmbeddingDiagnostics(
        embedding_id=vector.embedding_id,
        model_id=vector.model_identity,
        model_version=model_version,
        dimension=vector.dimension,
        normalization=normalization,
        configuration_hash=configuration_hash,
        vector_norm=vector.vector_norm,
        vector=list(vector.values) if include_vector else None,
        source_id=source_id,
        source_type=source_type,
        latency_ms=latency_ms,
    )
