"""Request-local configuration control for the supported evaluation architecture."""

from dataclasses import dataclass
from typing import Any

from llm_grounding.infrastructure.config.models import (
    ArchitectureConfig,
    RerankerConfig,
)
from llm_grounding.ingestion.models import EmbeddingConfig, IngestionConfig
from llm_grounding.observability.configuration import build_effective_configuration
from llm_grounding.observability.models import EffectiveConfiguration

SUPPORTED_OVERRIDE_FIELDS = (
    "parser",
    "chunker",
    "embedder",
    "embedding",
    "vector_retriever",
    "top_k",
    "retrieval_top_k",
    "rerank_top_k",
    "reranker",
    "reranker_model_name",
    "reranker_device",
    "reranker_batch_size",
    "context_chars",
    "summary_per_facet",
    "prompt_id",
    "model_name",
)


@dataclass(frozen=True)
class EvaluationConfig:
    """Validated request-local values used by an evaluation run."""

    overrides: dict[str, Any]
    effective: EffectiveConfiguration


class EvaluationConfigService:
    """Validate and materialize Basic RAG overrides without global mutation."""

    def validate(
        self,
        architecture: ArchitectureConfig,
        ingestion: IngestionConfig,
        overrides: dict[str, Any] | None,
        *,
        corpus_id: str | None = None,
    ) -> EvaluationConfig:
        """Return an immutable effective configuration for one request.

        Only components implemented by ``basic_rag_v1`` are accepted. The
        returned Pydantic configuration is built from copies and never changes
        the application composition root, so simultaneous evaluations remain
        isolated from one another.
        """
        values = dict(overrides or {})
        unsupported = sorted(set(values) - set(SUPPORTED_OVERRIDE_FIELDS))
        if unsupported:
            raise ValueError(
                "Unsupported evaluation configuration override(s): "
                + ", ".join(unsupported)
            )
        self._validate_values(values)
        if architecture.name != "basic_rag_v1":
            raise ValueError("Evaluation configuration requires basic_rag_v1")
        if architecture.retriever.type != "vector":
            raise ValueError("Evaluation configuration requires the vector retriever")
        architecture_copy = architecture.model_copy(deep=True)
        ingestion_copy = ingestion.model_copy(deep=True)
        architecture_copy, ingestion_copy = self._apply(
            architecture_copy, ingestion_copy, values
        )
        effective = build_effective_configuration(
            architecture_copy,
            ingestion_copy,
            corpus_id=corpus_id,
        )
        return EvaluationConfig(values, effective)

    @staticmethod
    def _validate_values(values: dict[str, Any]) -> None:
        """Validate override value types and bounded runtime settings."""
        text_fields = (
            "parser",
            "chunker",
            "embedder",
            "vector_retriever",
            "prompt_id",
            "model_name",
            "reranker",
            "reranker_model_name",
            "reranker_device",
        )
        for field in text_fields:
            if field in values and (
                not isinstance(values[field], str) or not values[field].strip()
            ):
                raise ValueError(f"{field} must be a non-empty string")
        if "embedding" in values and not isinstance(
            values["embedding"], (dict, EmbeddingConfig)
        ):
            raise ValueError("embedding must be an object")
        integer_bounds = {
            "top_k": (1, 100),
            "retrieval_top_k": (1, 100),
            "rerank_top_k": (1, 100),
            "context_chars": (1, 200_000),
            "summary_per_facet": (1, 20),
            "reranker_batch_size": (1, 512),
        }
        for field, (minimum, maximum) in integer_bounds.items():
            if field in values and (
                isinstance(values[field], bool)
                or not isinstance(values[field], int)
                or not minimum <= values[field] <= maximum
            ):
                raise ValueError(
                    f"{field} must be an integer in [{minimum}, {maximum}]"
                )

    @staticmethod
    def _apply(
        architecture: ArchitectureConfig,
        ingestion: IngestionConfig,
        values: dict[str, Any],
    ) -> tuple[ArchitectureConfig, IngestionConfig]:
        """Apply validated scalar values to request-local configuration copies."""
        expected = {
            "parser": "pymupdf",
            "chunker": ("fixed_window", "block_aware"),
            "embedder": ("sentence_transformers", "fake"),
            "vector_retriever": "vector",
            "reranker": ("identity", "top_k", "cross_encoder"),
        }
        for field, accepted in expected.items():
            if field in values and values[field] not in (
                accepted if isinstance(accepted, tuple) else (accepted,)
            ):
                raise ValueError(f"Unsupported {field}: {values[field]}")
        if "parser" in values:
            ingestion = ingestion.model_copy(update={"parser": values["parser"]})
        if "chunker" in values:
            ingestion = ingestion.model_copy(
                update={
                    "chunker": ingestion.chunker.model_copy(
                        update={"type": values["chunker"]}
                    )
                }
            )
        if "embedder" in values:
            ingestion = ingestion.model_copy(
                update={
                    "embedding": ingestion.embedding.model_copy(
                        update={"type": values["embedder"]}
                    )
                }
            )
        if "embedding" in values:
            embedding_value = values["embedding"]
            if isinstance(embedding_value, dict):
                embedding = IngestionConfig.model_validate(
                    {
                        **ingestion.model_dump(mode="json"),
                        "embedding": {
                            **ingestion.embedding.model_dump(mode="json"),
                            **embedding_value,
                        },
                    }
                ).embedding
            else:
                embedding = embedding_value
            ingestion = ingestion.model_copy(update={"embedding": embedding})
        retriever = architecture.retriever
        updates: dict[str, Any] = {
            key: values[key]
            for key in ("top_k", "context_chars", "summary_per_facet")
            if key in values
        }
        architecture = architecture.model_copy(
            update={"retriever": retriever.model_copy(update=updates)}
        )
        reranker_updates: dict[str, Any] = {}
        if "retrieval_top_k" in values:
            reranker_updates["retrieval_top_k"] = values["retrieval_top_k"]
        for source, target in (
            ("rerank_top_k", "rerank_top_k"),
            ("reranker_model_name", "model_name"),
            ("reranker_device", "device"),
            ("reranker_batch_size", "batch_size"),
        ):
            if source in values:
                reranker_updates[target] = values[source]
        if "reranker" in values:
            reranker_updates["type"] = values["reranker"]
        if reranker_updates:
            reranker = RerankerConfig.model_validate(
                {**architecture.reranker.model_dump(mode="json"), **reranker_updates}
            )
            architecture = architecture.model_copy(update={"reranker": reranker})
        if "prompt_id" in values:
            architecture = architecture.model_copy(
                update={
                    "reasoner": architecture.reasoner.model_copy(
                        update={"prompt_id": values["prompt_id"]}
                    )
                }
            )
        if "model_name" in values:
            architecture = architecture.model_copy(
                update={
                    "reasoner": architecture.reasoner.model_copy(
                        update={
                            "model": architecture.reasoner.model.model_copy(
                                update={"name": values["model_name"]}
                            )
                        }
                    )
                }
            )
        return architecture, ingestion
