"""Central resolver for standalone model and component configurations.

The registry is the boundary between declarative YAML and executable runtime
schemas. It owns ID uniqueness and reference resolution, while the existing
component registry remains responsible for implementation availability.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel, TypeAdapter, ValidationError

from llm_grounding.infrastructure.config.declarative import (
    ComponentDefinition,
    DeclarativePipelineConfig,
    FakeModelDefinition,
    ModelDefinition,
    OpenAICompatibleModelDefinition,
    SentenceTransformersModelDefinition,
)
from llm_grounding.infrastructure.config.models import (
    ApplicationDefaults,
    ChatHistoryConfig,
    ComponentConfig,
    ModelCatalog,
    ModelConfig,
    PipelineConfig,
    PipelineEmbeddingConfig,
    PipelineQueryConfig,
    PipelineReasonerConfig,
    PipelineRepresentationConfig,
    RerankerConfig,
    RetrieverConfig,
    VectorBackendConfig,
)
from llm_grounding.ingestion.models import ChunkerConfig, EmbeddingConfig

_MODEL_ADAPTER: TypeAdapter[ModelDefinition] = TypeAdapter(ModelDefinition)
_ModelT = TypeVar("_ModelT", bound=BaseModel)


@dataclass(frozen=True)
class ConfigurationRegistry:
    """Resolve unique model/component IDs into established runtime schemas."""

    models: dict[str, ModelDefinition]
    components: dict[str, tuple[str, ComponentDefinition]]
    defaults: ApplicationDefaults = ApplicationDefaults()

    @classmethod
    def from_directories(
        cls,
        models_directory: Path,
        components_directory: Path,
        *,
        defaults: ApplicationDefaults | None = None,
    ) -> "ConfigurationRegistry":
        """Load and validate all standalone YAML definitions in deterministic order."""
        models: dict[str, ModelDefinition] = {}
        components: dict[str, tuple[str, ComponentDefinition]] = {}
        for path in _yaml_paths(models_directory):
            model_definition = _load_model(path)
            if model_definition.id in models:
                raise ValueError(
                    f"Duplicate model configuration ID: {model_definition.id}"
                )
            models[model_definition.id] = model_definition
        for path in _yaml_paths(components_directory):
            category = path.parent.name
            component_definition = _load_component(path)
            if component_definition.id in components:
                raise ValueError(
                    f"Duplicate component configuration ID: {component_definition.id}"
                )
            components[component_definition.id] = (category, component_definition)
        return cls(models, components, defaults or ApplicationDefaults())

    def model(self, model_id: str) -> ModelDefinition:
        """Return one model definition or raise an actionable reference error."""
        try:
            return self.models[model_id]
        except KeyError as exc:
            raise ValueError(f"Unknown model configuration ID: {model_id}") from exc

    def component(self, category: str, component_id: str) -> ComponentDefinition:
        """Return one category-matched component definition."""
        try:
            actual_category, definition = self.components[component_id]
        except KeyError as exc:
            raise ValueError(
                f"Unknown {category} component configuration ID: {component_id}"
            ) from exc
        if actual_category != category:
            raise ValueError(
                f"Component {component_id} belongs to {actual_category}, "
                f"not {category}"
            )
        return definition

    def model_catalog(self) -> ModelCatalog:
        """Convert standalone model definitions to the runtime catalog."""
        embedding: dict[str, EmbeddingConfig] = {}
        reasoning: dict[str, ModelConfig] = {}
        for model_id, definition in self.models.items():
            if definition.stage == "reasoner":
                reasoning[model_id] = self._model_config(model_id)
            elif definition.stage == "embedding":
                embedding[model_id] = self._embedding_config(model_id)
        return ModelCatalog(
            embedding_models=embedding,
            reasoning_models=reasoning,
        )

    def pipeline(
        self,
        source: DeclarativePipelineConfig,
        catalog: ModelCatalog | None = None,
    ) -> PipelineConfig:
        """Resolve one composition-only pipeline into the stable pipeline schema."""
        representation = None
        if source.representation is not None:
            model_catalog = catalog or self.model_catalog()
            embedding = source.representation.embedding
            vector_backend_id = source.representation.vector_backend
            if embedding is None and vector_backend_id is not None:
                raise ValueError(
                    "vector_backend requires an embedding configuration; set both "
                    "to null for PostgreSQL-only ingestion"
                )
            vector_backend = None
            if embedding is not None:
                embedding_config = model_catalog.embedding(
                    self._model_id_for_component("embedding", embedding)
                )
                vector_backend_id = vector_backend_id or self.defaults.vector_backend
                if vector_backend_id is None:
                    raise ValueError(
                        "An embedding configuration requires a vector_backend or "
                        "defaults.vector_backend"
                    )
                vector_backend = self._component(
                    "vector_backend", vector_backend_id, VectorBackendConfig
                )
                self._validate_backend_compatibility(
                    embedding_config, vector_backend
                )
            representation = PipelineRepresentationConfig(
                parser=self._component(
                    "parser", source.representation.parser, ComponentConfig
                ),
                knowledge_builder=(
                    self._component(
                        "knowledge_builder",
                        source.representation.knowledge_builder,
                        ComponentConfig,
                    )
                    if source.representation.knowledge_builder is not None
                    else None
                ),
                chunker=self._component(
                    "chunker", source.representation.chunker, ChunkerConfig
                ),
                embedding=(
                    PipelineEmbeddingConfig(
                        model_ref=self._model_id_for_component("embedding", embedding)
                    )
                    if embedding is not None
                    else None
                ),
                vector_backend=vector_backend,
            )
        query = None
        if source.query is not None:
            query = PipelineQueryConfig(
                retriever=self._component(
                    "retriever", source.query.retriever, RetrieverConfig
                ),
                reranker=self._reranker(source.query.reranker),
                reasoner=self._pipeline_reasoner(source.query.reasoner),
                assessors=tuple(
                    self._component("assessor", item, ComponentConfig)
                    for item in source.query.assessors
                ),
                decision_policy=self._component(
                    "decision_policy", source.query.decision_policy, ComponentConfig
                ),
                renderer=self._component(
                    "renderer", source.query.renderer, ComponentConfig
                ),
                chat_history=(
                    ChatHistoryConfig.model_validate(source.query.chat_history)
                    if source.query.chat_history is not None
                    else None
                ),
            )
        return PipelineConfig(
            schema_version=1,
            kind=source.kind,
            name=source.name,
            version=source.version,
            description=source.description,
            is_default=source.is_default,
            representation=representation,
            query=query,
        )

    @staticmethod
    def _validate_backend_compatibility(
        embedding: EmbeddingConfig, backend: VectorBackendConfig
    ) -> None:
        """Reject embedding representations unsupported by the selected backend."""
        if embedding.representation != backend.representation:
            raise ValueError(
                f"Vector backend {backend.type!r} does not support "
                f"{embedding.representation} embeddings"
            )
        if embedding.similarity_metric != backend.similarity_metric:
            raise ValueError(
                f"Vector backend {backend.type!r} does not support "
                f"{embedding.similarity_metric} similarity"
            )

    def _component(
        self,
        category: str,
        component_id: str,
        model: type[_ModelT],
    ) -> _ModelT:
        """Validate one component's operational settings with its runtime schema."""
        definition = self.component(category, component_id)
        if category not in {"embedder", "embedding", "reasoner", "reranker"}:
            if definition.model_id is not None:
                raise ValueError(
                    f"{category} component {component_id} must not reference a model"
                )
        values = {"type": definition.type, **definition.settings()}
        try:
            return model.model_validate(values)
        except ValidationError as exc:
            raise ValueError(
                f"Invalid {category} component configuration ID: {component_id}"
            ) from exc

    def _model_config(self, model_id: str) -> ModelConfig:
        """Resolve a standalone remote model to the existing LLM runtime schema."""
        definition = self.model(model_id)
        if not isinstance(definition, OpenAICompatibleModelDefinition):
            raise ValueError(
                f"Model {model_id} backend {definition.backend!r} "
                "cannot be used by a reasoning component"
            )
        return ModelConfig(
            provider=definition.provider,
            name=definition.name,
            base_url=definition.base_url,
            api_key_env=definition.auth.env_var,
            api_key_file_env=definition.auth.file_env_var,
            timeout_seconds=definition.timeout_seconds,
            retries=definition.retries,
        )

    def _embedding_config(self, model_id: str) -> EmbeddingConfig:
        """Resolve a standalone model to the existing ingestion embedder schema."""
        definition = self.model(model_id)
        if isinstance(definition, OpenAICompatibleModelDefinition):
            return EmbeddingConfig(
                type="openai_compatible",
                provider=definition.provider,
                model=definition.name,
                dimension=getattr(definition, "dimension", 384) or 384,
                base_url=definition.base_url,
                api_key_env=definition.auth.env_var,
                api_key_file_env=definition.auth.file_env_var,
                timeout_seconds=definition.timeout_seconds,
                retries=definition.retries,
            )
        if isinstance(definition, SentenceTransformersModelDefinition):
            return EmbeddingConfig(
                type="sentence_transformers",
                model=definition.name,
                revision=definition.revision,
                dimension=definition.dimension,
                normalize=definition.normalize,
                batch_size=definition.batch_size,
                cache_dir_env=definition.cache_dir_env,
            )
        if isinstance(definition, FakeModelDefinition):
            return EmbeddingConfig(
                type="fake",
                model=definition.name,
                dimension=definition.dimension,
                normalize=definition.normalize,
                batch_size=definition.batch_size,
            )
        raise ValueError(
            f"Model {model_id} backend {definition.backend!r} "
            "cannot be used as an ingestion embedder"
        )

    def _reranker(self, component_id: str) -> RerankerConfig:
        """Resolve a reranker component and substitute its model ID when needed."""
        definition = self.component("reranker", component_id)
        if definition.type == "cross_encoder" and definition.model_id is None:
            raise ValueError(f"Reranker component {component_id} requires model_id")
        if definition.type != "cross_encoder" and definition.model_id is not None:
            raise ValueError(
                f"Reranker component {component_id} does not accept model_id"
            )
        values = definition.settings()
        if definition.model_id is not None:
            model = self.model(definition.model_id)
            self._validate_model_stage(definition.model_id, model, "reranker")
            values["model_name"] = model.name
        return RerankerConfig.model_validate({"type": definition.type, **values})

    def _pipeline_reasoner(self, component_id: str) -> PipelineReasonerConfig:
        """Resolve a pipeline reasoner to an internal model ID reference."""
        definition = self.component("reasoner", component_id)
        if definition.model_id is None:
            raise ValueError(f"Reasoner component {component_id} requires model_id")
        return PipelineReasonerConfig.model_validate(
            {
                "type": definition.type,
                "model_ref": self._model_id_for_component("reasoner", component_id),
                **definition.settings(),
            }
        )

    def _model_id_for_component(self, category: str, component_id: str) -> str:
        """Resolve a model-bearing component to its model ID."""
        definition = self.component(category, component_id)
        if definition.model_id is None:
            raise ValueError(f"{category} component {component_id} requires model_id")
        model = self.model(definition.model_id)
        self._validate_model_stage(definition.model_id, model, category)
        return definition.model_id

    @staticmethod
    def _validate_model_stage(
        model_id: str,
        model: ModelDefinition,
        expected_stage: str,
    ) -> None:
        """Reject component references to models assigned to another stage."""
        if model.stage != expected_stage:
            raise ValueError(
                f"Model {model_id} is tagged for {model.stage}, "
                f"not {expected_stage}"
            )


def _yaml_paths(directory: Path) -> tuple[Path, ...]:
    """Discover YAML files recursively without making filenames authoritative."""
    if not directory.is_dir():
        raise ValueError(f"Configuration directory does not exist: {directory}")
    return tuple(sorted((*directory.rglob("*.yaml"), *directory.rglob("*.yml"))))


def _load_model(path: Path) -> ModelDefinition:
    """Load one provider-discriminated model definition with path context."""
    try:
        value = _MODEL_ADAPTER.validate_python(yaml.safe_load(path.read_text("utf-8")))
        return value
    except (OSError, UnicodeError, yaml.YAMLError, ValidationError) as exc:
        raise ValueError(f"Invalid model configuration file: {path}") from exc


def _load_component(path: Path) -> ComponentDefinition:
    """Load one reusable component definition with path context."""
    try:
        return ComponentDefinition.model_validate(
            yaml.safe_load(path.read_text("utf-8"))
        )
    except (OSError, UnicodeError, yaml.YAMLError, ValidationError) as exc:
        raise ValueError(f"Invalid component configuration file: {path}") from exc
