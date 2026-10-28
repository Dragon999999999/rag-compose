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
from llm_grounding.infrastructure.config.model_schema import validate_model_file
from llm_grounding.infrastructure.config.models import (
    ApplicationDefaults,
    AssessorConfig,
    ChatHistoryConfig,
    ComponentConfig,
    DecisionPolicyConfig,
    KnowledgeBuilderConfig,
    ModelCatalog,
    ModelConfig,
    PipelineConfig,
    PipelineEmbeddingConfig,
    PipelineExecutionConfig,
    PipelineQueryConfig,
    PipelineReasonerConfig,
    PipelineRepresentationConfig,
    RerankerConfig,
    RetrieverConfig,
    VectorBackendConfig,
)
from llm_grounding.infrastructure.config.schema import validate_component_file
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
            model_definition = _load_model(path, models_directory)
            if model_definition.id in models:
                raise ValueError(
                    f"Duplicate model configuration ID: {model_definition.id}"
                )
            models[model_definition.id] = model_definition
        for path in _yaml_paths(components_directory):
            category = path.parent.name
            component_definition = _load_component(path, components_directory)
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
                f"Component {component_id} belongs to {actual_category}, not {category}"
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
                self._validate_backend_compatibility(embedding_config, vector_backend)
            chunker = self._chunker(source.representation.chunker)
            knowledge_builder = (
                self._knowledge_builder(source.representation.knowledge_builder)
                if source.representation.knowledge_builder is not None
                else None
            )
            embedding_ref = (
                self._model_id_for_component("embedding", embedding)
                if embedding is not None
                else None
            )
            if chunker.implementation == "late_chunking":
                if embedding_ref != chunker.model_id:
                    raise ValueError(
                        "late_chunking model_id must match the ingestion embedding"
                    )
            representation = PipelineRepresentationConfig(
                parser=self._component(
                    "parser", source.representation.parser, ComponentConfig
                ),
                knowledge_builder=knowledge_builder,
                chunker=chunker,
                embedding=(
                    PipelineEmbeddingConfig(
                        model_ref=self._model_id_for_component("embedding", embedding),
                        component_id=embedding,
                    )
                    if embedding is not None
                    else None
                ),
                vector_backend=vector_backend,
            )
        query = None
        if source.query is not None:
            retriever = self._component(
                "retriever", source.query.retriever, RetrieverConfig
            )
            if retriever.query_embedder is not None:
                retriever = retriever.model_copy(
                    update={
                        "query_embedder": self._embedding_model_ref(
                            retriever.query_embedder
                        )
                    }
                )
            query = PipelineQueryConfig(
                retriever=retriever,
                reranker=self._reranker(source.query.reranker),
                reasoner=self._pipeline_reasoner(source.query.reasoner),
                assessors=tuple(
                    self._component("assessor", item, AssessorConfig)
                    for item in source.query.assessors
                ),
                decision_policy=self._component(
                    "decision_policy",
                    source.query.decision_policy,
                    DecisionPolicyConfig,
                ),
                renderer=self._component(
                    "renderer", source.query.renderer, ComponentConfig
                ),
                chat_history=(
                    ChatHistoryConfig.model_validate(source.query.chat_history)
                    if source.query.chat_history is not None
                    else None
                ),
                execution=PipelineExecutionConfig.model_validate(
                    source.query.execution or {}
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
            stage_overrides=source.stage_overrides,
        )

    @staticmethod
    def _validate_backend_compatibility(
        embedding: EmbeddingConfig, backend: VectorBackendConfig
    ) -> None:
        """Reject embedding representations unsupported by the selected backend."""
        if embedding.representation != backend.representation:
            raise ValueError(
                f"Vector backend {backend.implementation!r} does not support "
                f"{embedding.representation} embeddings"
            )
        if embedding.similarity_metric != backend.similarity_metric:
            raise ValueError(
                f"Vector backend {backend.implementation!r} does not support "
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
        if category not in {
            "embedding",
            "reasoner",
            "reranker",
            "chunker",
            "knowledge_builder",
        }:
            if definition.model_id is not None:
                raise ValueError(
                    f"{category} component {component_id} must not reference a model"
                )
        runtime_type = (
            definition.implementation
            if category in {"vector_backend", "reranker"}
            else definition.type
        )
        values = {
            "id": definition.id,
            "type": runtime_type,
            "implementation": definition.implementation,
            **definition.settings(),
        }
        if category in {"assessor", "decision_policy"}:
            values["input_terms"] = definition.input_terms
            values["output_terms"] = definition.output_terms
        if definition.requirements is not None:
            values["requirements"] = definition.requirements.model_dump(mode="json")
        try:
            return model.model_validate(values)
        except ValidationError as exc:
            raise ValueError(
                f"Invalid {category} component configuration ID: {component_id}"
            ) from exc

    def _chunker(self, component_id: str) -> ChunkerConfig:
        """Resolve a chunker and validate its optional model-stage reference."""
        definition = self.component("chunker", component_id)
        implementation = definition.implementation
        if implementation in {"semantic", "late_chunking", "llm_guided"}:
            if definition.model_id is None:
                raise ValueError(f"Chunker component {component_id} requires model_id")
            model = self.model(definition.model_id)
            expected = "embedding" if implementation != "llm_guided" else None
            if expected is not None:
                self._validate_model_stage(definition.model_id, model, expected)
            elif model.stage not in {"reasoner", "validation"}:
                raise ValueError(
                    f"Model {definition.model_id} is not a generation model"
                )
        return self._component("chunker", component_id, ChunkerConfig)

    def _knowledge_builder(
        self, component_id: str
    ) -> KnowledgeBuilderConfig:
        """Resolve builder settings and validate model-backed references."""
        definition = self.component("knowledge_builder", component_id)
        implementation = definition.implementation
        if implementation == "metadata_gliner":
            if definition.model_id is None:
                raise ValueError(
                    f"Knowledge builder {component_id} requires model_id"
                )
            model = self.model(definition.model_id)
            if model.stage != "validation" or model.location != "local":
                raise ValueError(
                    f"Model {definition.model_id} must be a local validation model"
                )
        elif definition.model_id is not None:
            raise ValueError(
                f"Knowledge builder {component_id} does not accept model_id"
            )
        return self._component(
            "knowledge_builder", component_id, KnowledgeBuilderConfig
        )

    def _model_config(self, model_id: str) -> ModelConfig:
        """Resolve a standalone remote model to the existing LLM runtime schema."""
        definition = self.model(model_id)
        if not isinstance(definition, OpenAICompatibleModelDefinition):
            raise ValueError(
                f"Model {model_id} backend {definition.backend!r} "
                "cannot be used by a reasoning component"
            )
        if definition.connection is None or definition.base_url is None:
            raise ValueError(f"Remote model {model_id} has no endpoint configuration")
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
                model_id=model_id,
                location=definition.location,
                type="openai_compatible",
                implementation="openai_compatible",
                provider=definition.provider,
                model=definition.name,
                embedding_space_id=definition.embedding_space_id,
                representation=definition.representation,
                similarity_metric=definition.similarity_metric,
                dimension=getattr(definition, "dimension", 384) or 384,
                base_url=definition.base_url,
                api_key_env=definition.auth.env_var,
                api_key_file_env=definition.auth.file_env_var,
                timeout_seconds=definition.timeout_seconds,
                retries=definition.retries,
            )
        if isinstance(definition, SentenceTransformersModelDefinition):
            return EmbeddingConfig(
                model_id=model_id,
                location=definition.location,
                type="sentence_transformers",
                implementation="sentence_transformers",
                model=definition.name,
                embedding_space_id=definition.embedding_space_id,
                representation=definition.representation,
                similarity_metric=definition.similarity_metric,
                revision=definition.revision,
                dimension=definition.dimension,
                normalize=definition.normalize,
                batch_size=definition.batch_size,
                cache_dir_env=definition.cache_dir_env,
                query_prefix=definition.query_prefix,
                document_prefix=definition.document_prefix,
                trust_remote_code=definition.trust_remote_code,
                late_chunking=bool(definition.parameter("late_chunking", False)),
            )
        if isinstance(definition, FakeModelDefinition):
            return EmbeddingConfig(
                model_id=model_id,
                location=definition.location,
                type="fake",
                implementation="fake",
                model=definition.name,
                embedding_space_id=definition.embedding_space_id,
                representation=definition.representation,
                similarity_metric=definition.similarity_metric,
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
        model_backed = definition.implementation in {"cross_encoder", "generative"}
        if model_backed and definition.model_id is None:
            raise ValueError(f"Reranker component {component_id} requires model_id")
        if not model_backed and definition.model_id is not None:
            raise ValueError(
                f"Reranker component {component_id} does not accept model_id"
            )
        values = definition.settings()
        values.pop("model_id", None)
        if definition.model_id is not None:
            model = self.model(definition.model_id)
            self._validate_model_stage(definition.model_id, model, "reranker")
            values["model_id"] = definition.model_id
            values["model_name"] = model.name
        return RerankerConfig.model_validate(
            {
                "id": definition.id,
                "type": definition.implementation,
                "implementation": definition.implementation,
                **values,
            }
        )

    def _pipeline_reasoner(self, component_id: str) -> PipelineReasonerConfig:
        """Resolve a pipeline reasoner to an internal model ID reference."""
        definition = self.component("reasoner", component_id)
        if definition.model_id is None:
            raise ValueError(f"Reasoner component {component_id} requires model_id")
        values = definition.settings()
        values.pop("model_id", None)
        return PipelineReasonerConfig.model_validate(
            {
                "id": definition.id,
                "type": definition.implementation,
                "implementation": definition.implementation,
                "model_ref": self._model_id_for_component("reasoner", component_id),
                **values,
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

    def _embedding_model_ref(self, reference: str) -> str:
        """Resolve either an embedding component ID or a catalog model ID."""
        if reference in self.models:
            self._validate_model_stage(reference, self.models[reference], "embedding")
            return reference
        return self._model_id_for_component("embedding", reference)

    @staticmethod
    def _validate_model_stage(
        model_id: str,
        model: ModelDefinition,
        expected_stage: str,
    ) -> None:
        """Reject component references to models assigned to another stage."""
        if model.stage != expected_stage:
            raise ValueError(
                f"Model {model_id} is tagged for {model.stage}, not {expected_stage}"
            )


def _yaml_paths(directory: Path) -> tuple[Path, ...]:
    """Discover component instances while excluding family schema files."""
    if not directory.is_dir():
        raise ValueError(f"Configuration directory does not exist: {directory}")
    return tuple(
        sorted(
            path
            for path in (*directory.rglob("*.yaml"), *directory.rglob("*.yml"))
            if path.name not in {"schema.yaml", "schema.yml"}
        )
    )


def _load_model(path: Path, models_directory: Path) -> ModelDefinition:
    """Load one provider-discriminated model definition with path context."""
    try:
        value = validate_model_file(path, models_directory)
        validated = _MODEL_ADAPTER.validate_python(value)
        return validated
    except (OSError, UnicodeError, yaml.YAMLError, ValidationError, ValueError) as exc:
        raise ValueError(f"Invalid model configuration file: {path}") from exc


def _load_component(path: Path, components_directory: Path) -> ComponentDefinition:
    """Validate and load one reusable component definition with path context."""
    try:
        value = validate_component_file(path, path.parent.name, components_directory)
        return ComponentDefinition.model_validate(value)
    except (OSError, UnicodeError, yaml.YAMLError, ValidationError, ValueError) as exc:
        raise ValueError(f"Invalid component configuration file: {path}") from exc
