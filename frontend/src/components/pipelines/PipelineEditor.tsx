import { useEffect, useMemo, useState } from "react";
import { useNavigate, useParams, useSearchParams } from "react-router-dom";

import type {
  CatalogStage,
  JsonSchema,
  ModelCatalog,
  Pipeline,
  PipelineCatalog,
  PipelineConfig,
} from "../../api/types";
import { Button, ErrorState } from "../common/ui";
import {
  useModelCatalog,
  usePipeline,
  usePipelineCatalog,
  usePipelineMutations,
  usePipelines,
} from "../../features/hooks";
import "./PipelinePages.css";

type StageKey =
  | "parser"
  | "knowledge_builder"
  | "chunker"
  | "embedding"
  | "vector_backend"
  | "retriever"
  | "reranker"
  | "reasoner"
  | "assessors"
  | "decision_policy"
  | "renderer";
type StageSection = "representation" | "query";
type StageDefinition = {
  key: StageKey;
  label: string;
  section: StageSection;
  catalogKey: string;
  description: string;
};
type DraftSnapshot = {
  config: string;
  name: string;
  description: string;
  version: string;
  isDefault: boolean;
};

const REPRESENTATION_STAGES: StageDefinition[] = [
  { key: "parser", label: "Parser", section: "representation", catalogKey: "parser", description: "Extracts source content into structured document blocks." },
  { key: "knowledge_builder", label: "Knowledge builder", section: "representation", catalogKey: "knowledge_builder", description: "Builds canonical knowledge units from parsed content." },
  { key: "chunker", label: "Chunker", section: "representation", catalogKey: "chunker", description: "Splits knowledge units into retrievable chunks." },
  { key: "embedding", label: "Embedding", section: "representation", catalogKey: "embedding", description: "Maps chunks into the vector representation used by retrieval." },
  { key: "vector_backend", label: "Vector backend", section: "representation", catalogKey: "vector_backend", description: "Stores and searches the generated vector index." },
];
const QUERY_STAGES: StageDefinition[] = [
  { key: "retriever", label: "Retriever", section: "query", catalogKey: "retriever", description: "Selects evidence relevant to a user question." },
  { key: "reranker", label: "Reranker", section: "query", catalogKey: "reranker", description: "Refines the ordering of retrieved evidence." },
  { key: "reasoner", label: "Reasoner", section: "query", catalogKey: "reasoner", description: "Generates a grounded interpretation from evidence." },
  { key: "assessors", label: "Assessors", section: "query", catalogKey: "assessor", description: "Evaluates evidence and answer properties." },
  { key: "decision_policy", label: "Decision policy", section: "query", catalogKey: "decision_policy", description: "Decides whether the grounded answer can be returned." },
  { key: "renderer", label: "Renderer", section: "query", catalogKey: "renderer", description: "Renders the validated answer for the client." },
];
const ALL_STAGES = [...REPRESENTATION_STAGES, ...QUERY_STAGES];

/** Edit complete RAG pipelines through a fixed lifecycle graph and catalog-backed inspector. */
export function PipelineEditor() {
  const { pipelineId } = useParams();
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const isNew = !pipelineId;
  const sourceId = pipelineId ?? searchParams.get("source") ?? undefined;
  const pipelines = usePipelines();
  const defaultId = pipelines.data?.find((pipeline) => pipeline.is_default)?.id;
  const source = usePipeline(
    sourceId ?? (searchParams.get("mode") === "default" ? defaultId : undefined),
  );
  const catalog = usePipelineCatalog();
  const models = useModelCatalog();
  const mutations = usePipelineMutations();
  const [draft, setDraft] = useState<PipelineConfig | null>(null);
  const [selectedStage, setSelectedStage] = useState<StageKey>("parser");
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [version, setVersion] = useState("1");
  const [isDefault, setIsDefault] = useState(false);
  const [loadedTemplate, setLoadedTemplate] = useState<string | null>(null);
  const [localError, setLocalError] = useState<string | null>(null);
  const [savedSnapshot, setSavedSnapshot] = useState<DraftSnapshot | null>(null);

  const template = source.data?.config
    ? { id: source.data.id, config: source.data.config, name: source.data.name }
    : null;
  const templateKey =
    template?.id ?? (searchParams.get("mode") === "blank" ? "blank" : null);

  useEffect(() => {
    if (!isNew && !template) return;
    if (!templateKey || loadedTemplate === templateKey) return;
    const initial = template ? cloneConfig(template.config) : emptyConfig();
    if (isNew && template) {
      initial.name = `Copy of ${template.name}`;
      initial.version = 1;
      initial.is_default = false;
    }
    setDraft(initial);
    setName(initial.name);
    setDescription(initial.description);
    setVersion(String(initial.version));
    setIsDefault(initial.is_default);
    setSavedSnapshot(snapshot(initial));
    setLoadedTemplate(templateKey);
  }, [isNew, loadedTemplate, template, templateKey]);

  useEffect(() => {
    if (!isNew || templateKey || loadedTemplate === "blank") return;
    const initial = emptyConfig();
    setDraft(initial);
    setName(initial.name);
    setDescription(initial.description);
    setVersion(String(initial.version));
    setIsDefault(false);
    setSavedSnapshot(snapshot(initial));
    setLoadedTemplate("blank");
  }, [isNew, loadedTemplate, templateKey]);

  const dirty = useMemo(() => {
    if (!draft || !savedSnapshot) return false;
    return (
      JSON.stringify(
        snapshotWithMetadata(draft, name, description, version, isDefault),
      ) !== JSON.stringify(savedSnapshot)
    );
  }, [description, draft, isDefault, name, savedSnapshot, version]);

  useEffect(() => {
    if (!dirty) return undefined;
    const warn = (event: BeforeUnloadEvent): void => {
      event.preventDefault();
      event.returnValue = "";
    };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [dirty]);

  const issues = useMemo(
    () => validateDraft(draft, name, version, catalog.data, models.data),
    [catalog.data, draft, models.data, name, version],
  );
  const selectedDefinition =
    ALL_STAGES.find((stage) => stage.key === selectedStage) ?? REPRESENTATION_STAGES[0];
  const selectedValue = draft ? readStage(draft, selectedDefinition) : undefined;
  const representationChanged = Boolean(
    draft &&
      savedSnapshot &&
      JSON.stringify(draft.representation) !==
        JSON.stringify(JSON.parse(savedSnapshot.config).representation),
  );
  const saving = mutations.create.isPending || mutations.update.isPending;
  const saveError = mutations.create.error ?? mutations.update.error;

  if (
    source.isLoading ||
    catalog.isLoading ||
    models.isLoading ||
    (isNew && searchParams.get("mode") === "default" && pipelines.isLoading)
  ) {
    return (
      <main className="pipeline-page">
        <div className="pipeline-loading">Loading pipeline editor…</div>
      </main>
    );
  }
  if (source.error && !isNew) {
    return (
      <main className="pipeline-page">
        <ErrorState error={source.error} retry={() => source.refetch()} />
      </main>
    );
  }
  if (catalog.error || models.error) {
    return (
      <main className="pipeline-page">
        <ErrorState
          error={(catalog.error ?? models.error)!}
          retry={() => {
            void catalog.refetch();
            void models.refetch();
          }}
        />
      </main>
    );
  }
  if (!draft || !catalog.data || !models.data) {
    return (
      <main className="pipeline-page">
        <ErrorState error={new Error("The pipeline editor could not initialize a configuration.")} />
      </main>
    );
  }

  const leave = (): void => {
    if (!dirty || window.confirm("Discard unsaved pipeline changes?")) {
      navigate("/pipelines");
    }
  };
  const save = (): void => {
    if (issues.length > 0) {
      setLocalError("Complete the required stages and fields before saving.");
      return;
    }
    const config: PipelineConfig = {
      ...draft,
      name: name.trim(),
      description,
      version: Number(version),
      is_default: isDefault,
    };
    setLocalError(null);
    const onSuccess = (saved: Pipeline): void => {
      setSavedSnapshot(
        snapshotWithMetadata(
          config,
          saved.name,
          saved.description,
          String(saved.version),
          saved.is_default,
        ),
      );
      setDraft(config);
      navigate("/pipelines");
    };
    if (pipelineId) {
      mutations.update.mutate({ pipelineId, config }, { onSuccess });
    } else {
      mutations.create.mutate(config, { onSuccess });
    }
  };

  return (
    <main className="pipeline-page pipeline-editor-page">
      <header className="pipeline-page-header editor-page-header">
        <div>
          <button className="back-link back-button" onClick={leave} type="button">
            ← Pipelines{source.data ? ` / ${source.data.name}` : ""}
          </button>
          <h1>{pipelineId ? "Edit pipeline" : "Create pipeline"}</h1>
          <p>Choose a stage in the graph to inspect its registered implementations and parameters.</p>
        </div>
        <div className="editor-actions">
          {dirty && <span className="unsaved-indicator">Unsaved changes</span>}
          <Button variant="ghost" onClick={leave}>Discard</Button>
          <Button variant="primary" disabled={saving || issues.length > 0} onClick={save}>
            {saving ? "Saving…" : "Save pipeline"}
          </Button>
        </div>
      </header>

      <section className="editor-meta">
        <label>Name<input maxLength={255} onChange={(event) => setName(event.target.value)} value={name} /></label>
        <label>Description<input maxLength={4000} onChange={(event) => setDescription(event.target.value)} value={description} /></label>
        <label>Version<input min="1" onChange={(event) => setVersion(event.target.value)} type="number" value={version} /></label>
        <label className="editor-checkbox"><input checked={isDefault} onChange={(event) => setIsDefault(event.target.checked)} type="checkbox" /> Default pipeline</label>
      </section>

      {(localError || saveError) && <p className="form-error pipeline-editor-error" role="alert">{localError ?? saveError?.message}</p>}
      {issues.length > 0 && <div className="validation-summary" role="alert"><strong>Configuration needs attention</strong><ul>{issues.map((issue) => <li key={issue}>{issue}</li>)}</ul></div>}

      <div className="pipeline-editor-layout">
        <section className="pipeline-graph" aria-label="Pipeline graph">
          <GraphSection label="Representation / indexing" stages={REPRESENTATION_STAGES} selected={selectedStage} config={draft} onSelect={setSelectedStage} />
          <div className="graph-divider" />
          <GraphSection label="Query processing" stages={QUERY_STAGES} selected={selectedStage} config={draft} onSelect={setSelectedStage} />
          {representationChanged && <p className="compatibility-warning"><strong>Representation changed.</strong> Changing parser, chunking, embedding, or vector settings may require a compatible index before this pipeline can execute.</p>}
        </section>
        <StageInspector
          catalog={catalog.data}
          definition={selectedDefinition}
          models={models.data}
          onChange={(value) => setDraft((current) => current ? updateStage(current, selectedDefinition, value) : current)}
          onAddAssessor={() => setDraft((current) => current ? updateAssessors(current, [...readAssessors(current), defaultStageValue(QUERY_STAGES[3], catalog.data, models.data)]) : current)}
          onRemoveAssessor={(index) => setDraft((current) => current ? updateAssessors(current, readAssessors(current).filter((_, itemIndex) => itemIndex !== index)) : current)}
          onChangeAssessor={(index, value) => setDraft((current) => current ? updateAssessors(current, readAssessors(current).map((item, itemIndex) => itemIndex === index ? value : item)) : current)}
          value={selectedValue}
        />
      </div>
    </main>
  );
}

function GraphSection({ label, stages, selected, config, onSelect }: { label: string; stages: StageDefinition[]; selected: StageKey; config: PipelineConfig; onSelect: (stage: StageKey) => void }) {
  return (
    <section className="graph-section">
      <h2>{label}</h2>
      <div className="graph-row">
        {stages.map((stage, index) => <span className="graph-node-wrap" key={stage.key}><StageNode definition={stage} selected={selected === stage.key} configured={Boolean(readStage(config, stage))} onSelect={() => onSelect(stage.key)} />{index < stages.length - 1 && <span aria-hidden="true" className="graph-arrow">→</span>}</span>)}
      </div>
    </section>
  );
}

function StageNode({ definition, selected, configured, onSelect }: { definition: StageDefinition; selected: boolean; configured: boolean; onSelect: () => void }) {
  return <button className={`graph-node ${selected ? "selected" : ""} ${configured ? "configured" : "missing"}`} onClick={onSelect} type="button"><span>{definition.label}</span>{!configured && <small>Configure</small>}</button>;
}

function StageInspector({ catalog, definition, models, value, onChange, onAddAssessor, onRemoveAssessor, onChangeAssessor }: { catalog: PipelineCatalog; definition: StageDefinition; models: ModelCatalog; value: Record<string, unknown> | undefined; onChange: (value: Record<string, unknown>) => void; onAddAssessor: () => void; onRemoveAssessor: (index: number) => void; onChangeAssessor: (index: number, value: Record<string, unknown>) => void }) {
  const stage = catalog.stages[definition.catalogKey];
  const assessorValues = definition.key === "assessors" ? readAssessorValue(value) : [];
  return (
    <aside className="pipeline-inspector">
      <div className="inspector-heading"><span>{definition.section === "representation" ? "Representation / indexing" : "Query processing"}</span><h2>{definition.label}</h2><p>{definition.description}</p></div>
      {!stage ? <p className="form-error">This stage is not available in the backend catalog.</p> : definition.key === "assessors" ? (
        <div className="assessor-editor">{assessorValues.map((assessor, index) => <div className="assessor-item" key={index}><div className="assessor-item-heading"><strong>Assessor {index + 1}</strong>{assessorValues.length > 1 && <button onClick={() => onRemoveAssessor(index)} type="button">Remove</button>}</div><SchemaFields stage={stage} value={assessor} models={models} stageKey={definition.key} onChange={(next) => onChangeAssessor(index, next)} /></div>)}<Button onClick={onAddAssessor}>＋ Add assessor</Button></div>
      ) : (
        <SchemaFields stage={stage} value={value ?? {}} models={models} stageKey={definition.key} onChange={onChange} />
      )}
      {definition.key === "embedding" && <p className="inspector-warning">Embedding changes alter the representation identity and may require a new compatible index.</p>}
      <p className="inspector-catalog-note">Options and parameter constraints are provided by the backend catalog.</p>
    </aside>
  );
}

function SchemaFields({ stage, value, models, stageKey, onChange }: { stage: CatalogStage; value: Record<string, unknown>; models: ModelCatalog; stageKey: StageKey; onChange: (value: Record<string, unknown>) => void }) {
  const properties = stage.schema.properties ?? {};
  const required = new Set(stage.schema.required ?? []);
  return <div className="schema-fields">{Object.entries(properties).map(([field, fieldSchema]) => <SchemaField key={field} field={field} schema={fieldSchema} required={required.has(field)} value={value[field]} implementations={field === "type" ? stage.implementations : undefined} modelOptions={field === "model_ref" ? (stageKey === "embedding" ? Object.keys(models.embedding) : stageKey === "reasoner" ? Object.keys(models.reasoning) : undefined) : undefined} onChange={(next) => { const updated = { ...value }; if (next === undefined) delete updated[field]; else updated[field] = next; onChange(updated); }} />)}</div>;
}

function SchemaField({ field, schema, required, value, implementations, modelOptions, onChange }: { field: string; schema: JsonSchema; required: boolean; value: unknown; implementations?: string[]; modelOptions?: string[]; onChange: (value: unknown) => void }) {
  const resolved = resolveSchema(schema);
  const options = implementations ?? modelOptions ?? resolved.enum?.filter((item): item is string => typeof item === "string");
  const actualValue = value ?? resolved.default ?? (resolved.const !== undefined ? resolved.const : "");
  const optional = !required && (schema.anyOf?.some((item) => item.type === "null") ?? false);
  const selectOptions = resolved.const !== undefined ? [String(resolved.const)] : options;
  return <label className="schema-field"><span>{humanize(field)}{required && <em>Required</em>}</span>{schema.description && <small>{schema.description}</small>}{selectOptions && selectOptions.length > 0 ? <select disabled={resolved.const !== undefined} onChange={(event) => onChange(event.target.value)} value={String(actualValue)}>{!required && <option value="">Not set</option>}{selectOptions.map((option) => <option key={option} value={option}>{option}</option>)}</select> : resolved.type === "boolean" ? <input checked={Boolean(actualValue)} onChange={(event) => onChange(event.target.checked)} type="checkbox" /> : resolved.type === "integer" || resolved.type === "number" ? <input max={resolved.maximum} min={resolved.minimum} onChange={(event) => onChange(event.target.value === "" ? (optional ? null : 0) : Number(event.target.value))} type="number" value={actualValue === null || actualValue === undefined ? "" : String(actualValue)} /> : <input maxLength={resolved.maxLength} minLength={resolved.minLength} onChange={(event) => onChange(event.target.value || (optional ? null : event.target.value))} type="text" value={actualValue === null || actualValue === undefined ? "" : String(actualValue)} />}{resolved.minimum !== undefined || resolved.maximum !== undefined ? <small className="schema-constraint">{resolved.minimum !== undefined && `Min ${resolved.minimum}`} {resolved.maximum !== undefined && `Max ${resolved.maximum}`}</small> : null}</label>;
}

function resolveSchema(schema: JsonSchema): JsonSchema {
  return schema.anyOf?.find((item) => item.type !== "null") ?? schema;
}

function readStage(config: PipelineConfig, definition: StageDefinition): Record<string, unknown> | undefined {
  const section = config[definition.section];
  const value = section[definition.key];
  if (definition.key === "assessors") return readAssessorValue(value)[0];
  return isRecord(value) ? value : undefined;
}

function readAssessors(config: PipelineConfig): Record<string, unknown>[] {
  return readAssessorValue(config.query.assessors);
}

function readAssessorValue(value: unknown): Record<string, unknown>[] {
  if (isRecord(value)) return [value];
  if (!Array.isArray(value)) return [];
  return value.filter(isRecord);
}

function updateStage(config: PipelineConfig, definition: StageDefinition, value: Record<string, unknown>): PipelineConfig {
  const next = cloneConfig(config);
  if (definition.key === "assessors") next.query.assessors = [value];
  else next[definition.section] = { ...next[definition.section], [definition.key]: value };
  return next;
}

function updateAssessors(config: PipelineConfig, values: Record<string, unknown>[]): PipelineConfig {
  const next = cloneConfig(config);
  next.query.assessors = values;
  return next;
}

function defaultStageValue(definition: StageDefinition, catalog: PipelineCatalog, models: ModelCatalog): Record<string, unknown> {
  const stage = catalog.stages[definition.catalogKey];
  return Object.fromEntries(Object.entries(stage?.schema.properties ?? {}).map(([field, schema]) => {
    const resolved = resolveSchema(schema);
    if (field === "type") return [field, stage?.implementations[0] ?? ""];
    if (field === "model_ref") return [field, definition.key === "embedding" ? Object.keys(models.embedding)[0] ?? "" : Object.keys(models.reasoning)[0] ?? ""];
    if (resolved.const !== undefined) return [field, resolved.const];
    if (resolved.default !== undefined) return [field, resolved.default];
    return [field, schema.anyOf?.some((item) => item.type === "null") ? null : ""];
  }));
}

function validateDraft(config: PipelineConfig | null, name: string, version: string, catalog?: PipelineCatalog, models?: ModelCatalog): string[] {
  if (!config || !catalog || !models) return ["Pipeline configuration is still loading."];
  const issues: string[] = [];
  if (!name.trim()) issues.push("Pipeline name is required.");
  if (!/^\d+$/.test(version) || Number(version) < 1) issues.push("Version must be a positive integer.");
  for (const definition of ALL_STAGES) {
    const value = readStage(config, definition);
    if (!value) issues.push(`${definition.label} must be configured.`);
    const stage = catalog.stages[definition.catalogKey];
    if (value && stage) {
      const type = value.type;
      if (typeof type === "string" && stage.implementations.length > 0 && !stage.implementations.includes(type)) issues.push(`${definition.label} uses an unsupported implementation.`);
      if (definition.key === "embedding" && typeof value.model_ref === "string" && !models.embedding[value.model_ref]) issues.push("Embedding model reference is unavailable.");
      if (definition.key === "reasoner" && typeof value.model_ref === "string" && !models.reasoning[value.model_ref]) issues.push("Reasoning model reference is unavailable.");
    }
  }
  return [...new Set(issues)];
}

function emptyConfig(): PipelineConfig {
  return { schema_version: 1, name: "new_pipeline", version: 1, description: "", is_default: false, representation: {}, query: {} };
}

function cloneConfig(config: PipelineConfig): PipelineConfig {
  return JSON.parse(JSON.stringify(config)) as PipelineConfig;
}

function snapshot(config: PipelineConfig): DraftSnapshot {
  return snapshotWithMetadata(config, config.name, config.description, String(config.version), config.is_default);
}

function snapshotWithMetadata(config: PipelineConfig, name: string, description: string, version: string, isDefault: boolean): DraftSnapshot {
  return { config: JSON.stringify(config), name, description, version, isDefault };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function humanize(value: string): string {
  return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}
