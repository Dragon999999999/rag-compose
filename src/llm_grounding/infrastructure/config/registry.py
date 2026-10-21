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
    DeclarativeArchitectureConfig,
    DeclarativePipelineConfig,
    FakeModelDefinition,
    ModelDefinition,
    OpenAICompatibleModelDefinition,
    SentenceTransformersModelDefinition,
)
from llm_grounding.infrastructure.config.models import (
    ArchitectureConfig,
    ChatHistoryConfig,
    ComponentConfig,
    EmbedderConfig,
    LocalModelConfig,
    ModelCatalog,
    ModelConfig,
    PipelineConfig,
    PipelineEmbeddingConfig,
    PipelineQueryConfig,
    PipelineReasonerConfig,
    PipelineRepresentationConfig,
    ReasonerConfig,
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

    @classmethod
    def from_directories(
        cls,
        models_directory: Path,
        components_directory: Path,
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
        return cls(models, components)

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
        """Convert standalone model definitions to the legacy runtime catalog."""
        embedding: dict[str, EmbeddingConfig] = {}
        reasoning: dict[str, ModelConfig] = {}
        for model_id, definition in self.models.items():
            if isinstance(definition, OpenAICompatibleModelDefinition):
                runtime = self._model_config(model_id)
                reasoning[model_id] = runtime
                embedding[model_id] = self._embedding_config(model_id)
            elif isinstance(
                definition, (SentenceTransformersModelDefinition, FakeModelDefinition)
            ):
                embedding[model_id] = self._embedding_config(model_id)
        return ModelCatalog(
            embedding_models=embedding,
            reasoning_models=reasoning,
        )

    def architecture(self, source: DeclarativeArchitectureConfig) -> ArchitectureConfig:
        """Resolve one composition-only architecture into runtime configuration."""
        parser = self._component("parser", source.parser, ComponentConfig)
        knowledge = self._component(
            "knowledge_builder", source.knowledge_builder, ComponentConfig
        )
        retriever = self._component("retriever", source.retriever, RetrieverConfig)
        reranker = self._reranker(source.reranker)
        reasoner = self._reasoner(source.reasoner)
        embedder = (
            self._embedder(source.embedder) if source.embedder is not None else None
        )
        return ArchitectureConfig(
            name=source.name,
            version=source.version,
            parser=parser,
            knowledge_builder=knowledge,
            retriever=retriever,
            reranker=reranker,
            embedder=embedder,
            reasoner=reasoner,
            assessors=tuple(
                self._component("assessor", item, ComponentConfig)
                for item in source.assessors
            ),
            decision_policy=self._component(
                "decision_policy", source.decision_policy, ComponentConfig
            ),
            renderer=self._component("renderer", source.renderer, ComponentConfig),
            chat_history=(
                ChatHistoryConfig.model_validate(source.chat_history)
                if source.chat_history is not None
                else None
            ),
        )

    def pipeline(self, source: DeclarativePipelineConfig) -> PipelineConfig:
        """Resolve one composition-only pipeline into the stable pipeline schema."""
        representation = None
        if source.representation is not None:
            representation = PipelineRepresentationConfig(
                parser=self._component(
                    "parser", source.representation.parser, ComponentConfig
                ),
                knowledge_builder=self._component(
                    "knowledge_builder",
                    source.representation.knowledge_builder,
                    ComponentConfig,
                ),
                chunker=self._component(
                    "chunker", source.representation.chunker, ChunkerConfig
                ),
                embedding=PipelineEmbeddingConfig(
                    model_ref=self._model_id_for_component(
                        "embedding", source.representation.embedding
                    )
                ),
                vector_backend=self._component(
                    "vector_backend",
                    source.representation.vector_backend,
                    VectorBackendConfig,
                ),
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

    def _embedder(self, component_id: str) -> EmbedderConfig:
        """Resolve an architecture embedder component and its model reference."""
        definition = self.component("embedder", component_id)
        if definition.model_id is None:
            raise ValueError(f"Embedder component {component_id} requires model_id")
        definition_model = self.model(definition.model_id)
        if isinstance(definition_model, SentenceTransformersModelDefinition):
            model: ModelConfig | LocalModelConfig = LocalModelConfig(
                provider="sentence_transformers",
                name=definition_model.name,
                revision=definition_model.revision,
                dimension=definition_model.dimension,
                normalize=definition_model.normalize,
                batch_size=definition_model.batch_size,
                cache_dir_env=definition_model.cache_dir_env,
            )
        else:
            model = self._model_config(definition.model_id)
        values = {"type": definition.type, "model": model, **definition.settings()}
        return EmbedderConfig.model_validate(values)

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
            values["model_name"] = self.model(definition.model_id).name
        return RerankerConfig.model_validate({"type": definition.type, **values})

    def _reasoner(self, component_id: str) -> ReasonerConfig:
        """Resolve an architecture reasoner component and its model reference."""
        definition = self.component("reasoner", component_id)
        if definition.model_id is None:
            raise ValueError(f"Reasoner component {component_id} requires model_id")
        return ReasonerConfig.model_validate(
            {
                "type": definition.type,
                "model": self._model_config(definition.model_id),
                **definition.settings(),
            }
        )

    def _pipeline_reasoner(self, component_id: str) -> PipelineReasonerConfig:
        """Resolve a pipeline reasoner to an internal model ID reference."""
        definition = self.component("reasoner", component_id)
        if definition.model_id is None:
            raise ValueError(f"Reasoner component {component_id} requires model_id")
        return PipelineReasonerConfig.model_validate(
            {
                "type": definition.type,
                "model_ref": definition.model_id,
                **definition.settings(),
            }
        )

    def _model_id_for_component(self, category: str, component_id: str) -> str:
        """Resolve a model-bearing component to its model ID."""
        definition = self.component(category, component_id)
        if definition.model_id is None:
            raise ValueError(f"{category} component {component_id} requires model_id")
        self.model(definition.model_id)
        return definition.model_id


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
