"""Controlled pipeline, chat, and request configuration override resolution."""

from collections.abc import Iterable, Mapping
from importlib.resources import files
from pathlib import Path

import yaml

from llm_grounding.infrastructure.config.models import (
    ModelCatalog,
    PipelineConfig,
    PipelineQueryConfig,
)

OverrideMap = Mapping[str, Mapping[str, object]]


class OverrideValidationError(ValueError):
    """Report an unknown, invalid, or out-of-scope runtime override."""


def catalog_for_query_overrides(
    config: PipelineConfig,
    fallback: ModelCatalog,
    layers: Iterable[tuple[str, OverrideMap]],
) -> ModelCatalog:
    """Extend a persisted model snapshot with models named by transient overrides."""
    catalog = config.effective_catalog(fallback)
    embedding_models = dict(catalog.embedding_models)
    for _, overrides in layers:
        retriever_overrides = overrides.get("retriever", {})
        if not isinstance(retriever_overrides, Mapping):
            continue
        query_embedder = retriever_overrides.get("query_embedder")
        if isinstance(query_embedder, str) and query_embedder not in embedding_models:
            embedding_models[query_embedder] = fallback.embedding(query_embedder)
    return catalog.model_copy(update={"embedding_models": embedding_models})


def apply_query_overrides(
    config: PipelineConfig,
    layers: Iterable[tuple[str, OverrideMap]],
) -> tuple[PipelineConfig, dict[str, object]]:
    """Apply controlled overrides in increasing precedence order.

    Component definitions are immutable. Each layer creates validated Pydantic
    copies, and only parameters annotated by the component JSON Schema can be
    changed at that scope.
    """
    if config.query is None:
        if any(overrides for _, overrides in layers):
            raise OverrideValidationError("Only query pipelines accept stage overrides")
        return config, {}
    query = config.query
    provenance: dict[str, object] = {}
    for scope, overrides in layers:
        query, applied = _apply_layer(query, scope, overrides)
        if applied:
            provenance[scope] = applied
    return config.model_copy(update={"query": query}), provenance


def _apply_layer(
    query: PipelineQueryConfig,
    scope: str,
    overrides: OverrideMap,
) -> tuple[PipelineQueryConfig, dict[str, object]]:
    """Apply one named override scope to the query-stage models."""
    if scope not in {"pipeline", "chat", "request"}:
        raise OverrideValidationError(f"Unknown override scope: {scope}")
    updated = query
    applied: dict[str, object] = {}
    stages = {
        "retriever": (query.retriever, type(query.retriever), "retriever"),
        "reranker": (query.reranker, type(query.reranker), "reranker"),
        "reasoner": (query.reasoner, type(query.reasoner), "reasoner"),
    }
    for stage, values in overrides.items():
        if stage not in stages:
            raise OverrideValidationError(f"Unknown query stage override: {stage}")
        if not isinstance(values, Mapping):
            raise OverrideValidationError(f"Overrides for {stage} must be an object")
        target, model, family = stages[stage]
        data = target.model_dump(mode="json")
        changes: dict[str, object] = {}
        for parameter, value in values.items():
            if parameter not in model.model_fields:
                raise OverrideValidationError(
                    f"Unknown {stage} override parameter: {parameter}"
                )
            if parameter in {
                "id",
                "type",
                "implementation",
                "requirements",
                "model",
                "model_ref",
                "provider",
            }:
                raise OverrideValidationError(
                    f"{stage}.{parameter} is immutable and cannot be overridden"
                )
            implementation = getattr(target, "implementation", None) or getattr(
                target, "type", ""
            )
            if not isinstance(implementation, str):
                raise OverrideValidationError(
                    f"{stage} has no registered implementation"
                )
            allowed = override_scopes(family, implementation, parameter)
            if scope not in allowed:
                raise OverrideValidationError(
                    f"{stage}.{parameter} cannot be overridden at {scope} scope"
                )
            data[parameter] = value
            changes[parameter] = value
        try:
            replacement = model.model_validate(data)
        except ValueError as exc:
            raise OverrideValidationError(
                f"Invalid {stage} override configuration"
            ) from exc
        if stage == "retriever":
            updated = updated.model_copy(update={"retriever": replacement})
        elif stage == "reranker":
            updated = updated.model_copy(update={"reranker": replacement})
        else:
            updated = updated.model_copy(update={"reasoner": replacement})
        applied[stage] = changes
    return updated, applied


def override_scopes(
    family: str,
    implementation: str,
    parameter: str,
) -> frozenset[str]:
    """Read the declared ``x-override-scopes`` annotation from a family schema."""
    schema_path = _schema_path(family)
    schema = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    config_properties = schema["allOf"][1]["properties"]["config"]["properties"]
    definition = config_properties.get(parameter)
    if not isinstance(definition, dict):
        return frozenset()
    scopes = definition.get("x-override-scopes", ())
    if not isinstance(scopes, list) or not all(
        isinstance(item, str) for item in scopes
    ):
        return frozenset()
    return frozenset(scopes)


def _schema_path(family: str) -> Path:
    """Find a component-family schema in a checkout or packaged installation."""
    checkout = Path(__file__).resolve().parents[4] / "config/components" / family
    if (checkout / "schema.yaml").is_file():
        return checkout / "schema.yaml"
    packaged = (
        files("llm_grounding.infrastructure.config")
        / "standalone/components"
        / family
        / "schema.yaml"
    )
    return Path(str(packaged))
