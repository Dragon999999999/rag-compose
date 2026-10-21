import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate, useParams, useSearchParams } from "react-router-dom";

import type { PipelineConfig } from "../../api/types";
import { Button, ErrorState } from "../common/ui";
import { usePipeline, usePipelineMutations, usePipelines } from "../../features/hooks";
import "./PipelinePages.css";

/** Edit a complete pipeline document while keeping validation in the backend. */
export function PipelineEditor() {
  const { pipelineId } = useParams();
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const isNew = !pipelineId;
  const sourceId = pipelineId ?? searchParams.get("source") ?? (
    searchParams.get("mode") === "default"
      ? undefined
      : undefined
  );
  const pipelines = usePipelines();
  const defaultId = pipelines.data?.find((pipeline) => pipeline.is_default)?.id;
  const source = usePipeline(sourceId ?? (searchParams.get("mode") === "default" ? defaultId : undefined));
  const mutations = usePipelineMutations();
  const [configText, setConfigText] = useState("");
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [version, setVersion] = useState("1");
  const [isDefault, setIsDefault] = useState(false);
  const [loadedSource, setLoadedSource] = useState<string | null>(null);
  const [parseError, setParseError] = useState<string | null>(null);

  useEffect(() => {
    if (!source.data?.config || loadedSource === source.data.id) return;
    const config = { ...source.data.config };
    if (isNew) {
      config.name = `Copy of ${source.data.name}`;
      config.version = 1;
      config.is_default = false;
    }
    setConfigText(JSON.stringify(config, null, 2));
    setName(config.name);
    setDescription(config.description);
    setVersion(String(config.version));
    setIsDefault(config.is_default);
    setLoadedSource(source.data.id);
  }, [isNew, loadedSource, source.data]);

  useEffect(() => {
    if (isNew && searchParams.get("mode") === "blank" && !configText) {
      const blank = blankConfig();
      setConfigText(JSON.stringify(blank, null, 2));
      setName(blank.name);
      setDescription(blank.description);
      setVersion(String(blank.version));
      setIsDefault(false);
      setLoadedSource("blank");
    }
  }, [configText, isNew, searchParams]);

  const parsedConfig = useMemo(() => parseConfig(configText), [configText]);
  const saving = mutations.create.isPending || mutations.update.isPending;
  const error = mutations.create.error ?? mutations.update.error;
  const stageEntries = stageSummary(parsedConfig);

  if (!isNew && source.isLoading) return <main className="pipeline-page"><div className="pipeline-loading">Loading pipeline…</div></main>;
  if (!isNew && source.error) return <main className="pipeline-page"><ErrorState error={source.error} retry={() => source.refetch()} /></main>;
  if (isNew && searchParams.get("mode") === "default" && pipelines.isLoading) return <main className="pipeline-page"><div className="pipeline-loading">Loading default pipeline…</div></main>;

  const save = (): void => {
    const parsed = parseConfig(configText);
    if (!parsed) {
      setParseError("Configuration must be valid JSON before it can be saved.");
      return;
    }
    if (!name.trim()) {
      setParseError("Pipeline name is required.");
      return;
    }
    const config: PipelineConfig = {
      ...parsed,
      name: name.trim(),
      description,
      version: Number(version),
      is_default: isDefault,
    };
    setParseError(null);
    if (pipelineId) {
      mutations.update.mutate({ pipelineId, config }, { onSuccess: () => navigate("/pipelines") });
    } else {
      mutations.create.mutate(config, { onSuccess: () => navigate("/pipelines") });
    }
  };

  return (
    <main className="pipeline-page pipeline-editor-page">
      <header className="pipeline-page-header">
        <div>
          <Link className="back-link" to="/pipelines">← Pipeline library</Link>
          <h1>{pipelineId ? "Edit pipeline" : "Create pipeline"}</h1>
          <p>Configure the registered stages and let the backend validate compatibility.</p>
        </div>
        <div className="editor-actions">
          <Button variant="ghost" onClick={() => navigate("/pipelines")}>Cancel</Button>
          <Button variant="primary" disabled={saving} onClick={save}>{saving ? "Saving…" : "Save pipeline"}</Button>
        </div>
      </header>

      <section className="editor-meta">
        <label>Name<input maxLength={255} onChange={(event) => setName(event.target.value)} value={name} /></label>
        <label>Description<input maxLength={4000} onChange={(event) => setDescription(event.target.value)} value={description} /></label>
        <label>Version<input min="1" onChange={(event) => setVersion(event.target.value)} type="number" value={version} /></label>
        <label className="editor-checkbox"><input checked={isDefault} onChange={(event) => setIsDefault(event.target.checked)} type="checkbox" /> Default pipeline</label>
      </section>

      {parseError && <p className="form-error" role="alert">{parseError}</p>}
      {error && <p className="form-error" role="alert">{error.message}</p>}

      <section className="pipeline-stage-strip" aria-label="Pipeline stages">
        {stageEntries.map(([label, value]) => <div className="pipeline-stage" key={label}><span>{label}</span><strong>{value}</strong></div>)}
      </section>

      <section className="pipeline-config-editor">
        <div className="config-editor-heading"><div><h2>Configuration document</h2><p>Use implementation names and model references from the backend catalog.</p></div><span className={parsedConfig ? "config-valid" : "config-invalid"}>{parsedConfig ? "JSON valid" : "Needs attention"}</span></div>
        <textarea aria-label="Pipeline configuration JSON" onChange={(event) => { setConfigText(event.target.value); setParseError(null); }} spellCheck={false} value={configText} />
      </section>
    </main>
  );
}

function parseConfig(value: string): PipelineConfig | null {
  try {
    const parsed: unknown = JSON.parse(value);
    return parsed && typeof parsed === "object" ? parsed as PipelineConfig : null;
  } catch {
    return null;
  }
}

function blankConfig(): PipelineConfig {
  return {
    schema_version: 1,
    name: "new_pipeline",
    version: 1,
    description: "",
    is_default: false,
    representation: {},
    query: {},
  };
}

function stageSummary(config: PipelineConfig | null): [string, string][] {
  const representation = config?.representation ?? {};
  const query = config?.query ?? {};
  const value = (section: Record<string, unknown>, key: string): string => {
    const item = section[key];
    if (!item || typeof item !== "object") return "Not configured";
    const type = (item as Record<string, unknown>).type;
    const modelRef = (item as Record<string, unknown>).model_ref;
    if (typeof type === "string" && typeof modelRef === "string") return `${type} · ${modelRef}`;
    if (typeof type === "string") return type;
    if (typeof modelRef === "string") return modelRef;
    return "Configured";
  };
  return [
    ["Parser", value(representation, "parser")],
    ["Builder", value(representation, "knowledge_builder")],
    ["Chunker", value(representation, "chunker")],
    ["Embedding", value(representation, "embedding")],
    ["Vector", value(representation, "vector_backend")],
    ["Retriever", value(query, "retriever")],
    ["Reranker", value(query, "reranker")],
    ["Reasoner", value(query, "reasoner")],
    ["Policy", value(query, "decision_policy")],
    ["Renderer", value(query, "renderer")],
  ];
}
