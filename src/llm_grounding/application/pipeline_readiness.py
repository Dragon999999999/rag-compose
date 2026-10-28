"""Log the validated stages selected for project and chat pipelines."""

import logging

from llm_grounding.domain.corpus import Pipeline
from llm_grounding.infrastructure.config.models import PipelineConfig
from llm_grounding.ingestion.models import IngestionConfig

_LOGGER = logging.getLogger("llm_grounding.pipeline_readiness")


def log_ingestion_pipeline_ready(
    pipeline: Pipeline, config: IngestionConfig
) -> None:
    """Log every effective ingestion stage after project pipeline resolution."""
    if not isinstance(config, IngestionConfig):
        _LOGGER.warning(
            "Ingestion pipeline %s (%s) resolved without a readable configuration",
            pipeline.name,
            pipeline.id,
        )
        return
    stages: list[tuple[str, str]] = [
        ("parser", config.parser),
        ("knowledge builder", config.knowledge_builder or "disabled"),
        ("chunker", config.chunker.implementation),
    ]
    if config.embedding is not None:
        stages.append(("embedder", config.embedding.model_id or config.embedding.model))
    if config.vector_backend is not None:
        stages.append(("vector backend", config.vector_backend))
    _log_stages(pipeline, stages)


def log_query_pipeline_ready(pipeline: Pipeline) -> None:
    """Log every configured query stage after chat pipeline validation."""
    config = PipelineConfig.model_validate(pipeline.config)
    if config.query is None:
        return
    stages: list[tuple[str, str]] = [
        ("retriever", _component_name(config.query.retriever)),
        ("reranker", _component_name(config.query.reranker)),
        ("decision policy", _component_name(config.query.decision_policy)),
        ("reasoner", _component_name(config.query.reasoner)),
        ("renderer", _component_name(config.query.renderer)),
    ]
    stages.extend(
        ("assessor", _component_name(assessor))
        for assessor in config.query.assessors
    )
    _log_stages(pipeline, stages)


def _component_name(component: object) -> str:
    """Return the configured implementation name for one pipeline component."""
    implementation = getattr(component, "implementation", None)
    return str(implementation or getattr(component, "type", component))


def _log_stages(pipeline: Pipeline, stages: list[tuple[str, str]]) -> None:
    """Write consistent readiness messages for one selected pipeline."""
    for stage, name in stages:
        _LOGGER.info(
            "Stage %s %s is ready for %s pipeline %s (%s)",
            stage,
            name,
            pipeline.kind,
            pipeline.name,
            pipeline.id,
        )
