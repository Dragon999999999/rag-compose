"""Validate the checked-in RAG benchmark matrix against the live registries."""

from pathlib import Path

import yaml

from llm_grounding.infrastructure.config.declarative import DeclarativePipelineConfig
from llm_grounding.infrastructure.config.loader import load_model_catalog
from llm_grounding.infrastructure.config.registry import ConfigurationRegistry
from llm_grounding.orchestration.registry import ComponentRegistry

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "benchmarks/rag_benchmark_manifest.yaml"
BENCHMARK_ROOT = ROOT / "config/pipelines/benchmarks"


def _manifest() -> list[dict[str, object]]:
    """Read the benchmark manifest without importing application services."""
    value = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    experiments = value["experiments"]
    assert isinstance(experiments, list)
    return experiments


def test_manifest_contains_all_76_experiments_and_stable_ids() -> None:
    """Every requested benchmark ID appears exactly once in the manifest."""
    experiments = _manifest()
    assert len(experiments) == 76
    ids = [experiment["id"] for experiment in experiments]
    assert len(set(ids)) == 76
    assert (
        sum(experiment["status"] == "supported" for experiment in experiments) == 75
    )
    unsupported = next(
        experiment for experiment in experiments if experiment["id"] == "CHUNK-08"
    )
    assert unsupported["status"] == "unsupported"
    assert isinstance(unsupported["reason"], str)


def test_supported_benchmark_yaml_uses_current_registries() -> None:
    """All supported benchmark YAML is valid against the actual runtime schemas."""
    catalog = load_model_catalog()
    configuration_registry = ConfigurationRegistry.from_directories(
        ROOT / "config/models", ROOT / "config/components"
    )
    component_registry = ComponentRegistry()
    supported_paths = {
        Path(experiment["query_pipeline"])
        for experiment in _manifest()
        if experiment["status"] == "supported"
    } | {
        Path(experiment["ingestion_pipeline"])
        for experiment in _manifest()
        if experiment["status"] == "supported"
    }

    assert supported_paths
    for relative_path in supported_paths:
        path = ROOT / relative_path
        source = DeclarativePipelineConfig.model_validate(
            yaml.safe_load(path.read_text(encoding="utf-8"))
        )
        config = configuration_registry.pipeline(source, catalog)
        component_registry.validate_pipeline(config, catalog)


def test_manifest_pipeline_references_are_existing_and_separated_by_kind() -> None:
    """Manifest references point to files of the expected ingestion/query kind."""
    for experiment in _manifest():
        if experiment["status"] != "supported":
            continue
        ingestion = ROOT / Path(experiment["ingestion_pipeline"])
        query = ROOT / Path(experiment["query_pipeline"])
        assert ingestion.is_file()
        assert query.is_file()
        assert (
            yaml.safe_load(ingestion.read_text(encoding="utf-8"))["kind"]
            == "ingestion"
        )
        assert yaml.safe_load(query.read_text(encoding="utf-8"))["kind"] == "query"
