"""Build canonical diagnostics for query and document embeddings."""

from llm_grounding.embedding.models import EmbeddingVector
from llm_grounding.observability.models import EmbeddingDiagnostics


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
