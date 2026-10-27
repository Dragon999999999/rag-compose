import { useEffect, useMemo, useState } from "react";
import { useNavigate, useParams, useSearchParams } from "react-router-dom";

import { ApiRequestError } from "../../api/client";
import type {
  CatalogStage,
  ExecutionStrategy,
  JsonSchema,
  ModelCatalog,
  Pipeline,
  PipelineCatalog,
  PipelineConfig,
  PipelineKind,
  PipelineUpdateImpact,
} from "../../api/types";
import { Dialog } from "../common/Dialog";
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
type BranchStart = "retriever" | "reranker" | "assessor";
type QueryBranch = {
  id: string;
  start_at: BranchStart;
  retriever?: string;
  reranker?: string;
  assessors?: string[];
};
type GraphNodeLayout = {
  key: string;
  definition: StageDefinition;
  x: number;
  y: number;
  shared: boolean;
  branchId?: string;
};
type GraphEdgeLayout = { from: string; to: string };
type BranchGraphLayout = {
  nodes: GraphNodeLayout[];
  edges: GraphEdgeLayout[];
  branches: Array<{ id: string; label: string; center: number; removable: boolean }>;
  width: number;
  height: number;
};

const GRAPH_NODE_WIDTH = 142;
const GRAPH_NODE_HEIGHT = 72;
const GRAPH_COLUMN_GAP = 36;
const GRAPH_ROW_GAP = 116;
const GRAPH_TOP_PADDING = 64;
const GRAPH_BOTTOM_PADDING = 76;

const REPRESENTATION_STAGES: StageDefinition[] = [
  { key: "parser", label: "Parser", section: "representation", catalogKey: "parser", description: "Extracts source content into structured document blocks." },
  { key: "knowledge_builder", label: "Knowledge builder", section: "representation", catalogKey: "knowledge_builder", description: "Builds canonical knowledge units from parsed content." },
  { key: "chunker", label: "Chunker", section: "representation", catalogKey: "chunker", description: "Splits knowledge units into retrievable chunks." },
  { key: "embedding", label: "Embedding", section: "representation", catalogKey: "embedding", description: "Maps chunks into the vector representation used by retrieval." },
  { key: "vector_backend", label: "Vector backend", section: "representation", catalogKey: "vector_backend", description: "Stores and searches the generated vector index." },
];
export const QUERY_STAGES: StageDefinition[] = [
  { key: "retriever", label: "Retriever", section: "query", catalogKey: "retriever", description: "Selects evidence relevant to a user question." },
  { key: "reranker", label: "Reranker", section: "query", catalogKey: "reranker", description: "Refines the ordering of retrieved evidence." },
  { key: "assessors", label: "Assessors", section: "query", catalogKey: "assessor", description: "Evaluates evidence and answer properties." },
  { key: "decision_policy", label: "Decision policy", section: "query", catalogKey: "decision_policy", description: "Decides whether the grounded answer can be returned." },
  { key: "reasoner", label: "Reasoner", section: "query", catalogKey: "reasoner", description: "Generates a grounded interpretation from evidence." },
  { key: "renderer", label: "Renderer", section: "query", catalogKey: "renderer", description: "Renders the validated answer for the client." },
];

/** Edit complete RAG pipelines through a fixed lifecycle graph and catalog-backed inspector. */
export function PipelineEditor({ onNotFound }: { onNotFound?: (message: string) => void }) {
  const { pipelineId } = useParams();
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const isNew = !pipelineId;
  const sourceId = pipelineId ?? searchParams.get("source") ?? undefined;
  const requestedKind = parsePipelineKind(searchParams.get("kind"));
  const pipelines = usePipelines(false, requestedKind);
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
  const [updateImpact, setUpdateImpact] = useState<PipelineUpdateImpact | null>(null);
  const pipelineKind: PipelineKind = source.data?.kind ?? source.data?.config?.kind ?? requestedKind ?? "query";
  const visibleStages = stagesForKind(pipelineKind);

  const template = source.data?.config
    ? { id: source.data.id, config: source.data.config, name: source.data.name }
    : null;
  const templateKey =
    template?.id ?? (searchParams.get("mode") === "blank" ? "blank" : null);

  useEffect(() => {
    if (!isNew && !template) return;
    if (!templateKey || loadedTemplate === templateKey) return;
    const initial = template ? cloneConfig(template.config) : emptyConfig(pipelineKind);
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
    setSelectedStage(stagesForKind(initial.kind)[0].key);
  }, [isNew, loadedTemplate, pipelineKind, template, templateKey]);

  useEffect(() => {
    if (!isNew || templateKey || loadedTemplate === "blank") return;
    const initial = emptyConfig(pipelineKind);
    setDraft(initial);
    setName(initial.name);
    setDescription(initial.description);
    setVersion(String(initial.version));
    setIsDefault(false);
    setSavedSnapshot(snapshot(initial));
    setLoadedTemplate("blank");
    setSelectedStage(stagesForKind(initial.kind)[0].key);
  }, [isNew, loadedTemplate, pipelineKind, templateKey]);

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

  useEffect(() => {
    if (
      isNew ||
      !(source.error instanceof ApiRequestError) ||
      source.error.status !== 404
    ) {
      return;
    }
    onNotFound?.(source.error.message);
    navigate("/pipelines", { replace: true });
  }, [isNew, navigate, onNotFound, source.error]);

  const issues = useMemo(
    () => validateDraft(draft, name, version, visibleStages, catalog.data, models.data),
    [catalog.data, draft, models.data, name, version, visibleStages],
  );
  const selectedDefinition =
    visibleStages.find((stage) => stage.key === selectedStage) ?? visibleStages[0];
  const selectedValue = draft ? readStage(draft, selectedDefinition) : undefined;
  const representationChanged = Boolean(
      draft &&
      savedSnapshot &&
      JSON.stringify(draft.representation ?? null) !==
        JSON.stringify(JSON.parse(savedSnapshot.config).representation ?? null),
  );
  const saving =
    mutations.create.isPending ||
    mutations.update.isPending ||
    mutations.impact.isPending;
  const saveError =
    mutations.create.error ?? mutations.update.error ?? mutations.impact.error;

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
    setUpdateImpact(null);
    const onSuccess = (saved: Pipeline): void => {
      const savedConfig = { ...config, version: saved.version };
      setSavedSnapshot(
        snapshotWithMetadata(
          savedConfig,
          saved.name,
          saved.description,
          String(saved.version),
          saved.is_default,
        ),
      );
      setDraft(savedConfig);
      setVersion(String(saved.version));
      navigate("/pipelines");
    };
    if (pipelineId) {
      mutations.impact.mutate(
        { pipelineId, config },
        {
          onSuccess: (impact) => {
            if (impact.conversation_count === 0 && (impact.project_count ?? 0) === 0) {
              mutations.update.mutate(
                {
                  pipelineId,
                  config,
                  expectedConfigHash: source.data?.config_hash,
                },
                { onSuccess },
              );
              return;
            }
            setUpdateImpact(impact);
          },
        },
      );
    } else {
      mutations.create.mutate(config, { onSuccess });
    }
  };

  const updateExisting = (): void => {
    if (!pipelineId || !updateImpact) return;
    const impact = updateImpact;
    const config: PipelineConfig = {
      ...draft,
      name: name.trim(),
      description,
      version: Number(version),
      is_default: isDefault,
    };
    setUpdateImpact(null);
    mutations.update.mutate(
      {
        pipelineId,
        config,
        expectedConfigHash: impact.current_config_hash,
        migrateConversations: true,
      },
      {
        onSuccess: (saved) => {
          const savedConfig = { ...config, version: saved.version };
          setSavedSnapshot(
            snapshotWithMetadata(
              savedConfig,
              saved.name,
              saved.description,
              String(saved.version),
              saved.is_default,
            ),
          );
          setDraft(savedConfig);
          setVersion(String(saved.version));
          navigate("/pipelines");
        },
        onError: (error) => setLocalError(error.message),
      },
    );
  };

  const createCopy = (): void => {
    if (!updateImpact) return;
    const config: PipelineConfig = {
      ...draft,
      name: `Copy of ${name.trim()}`,
      description,
      version: 1,
      is_default: false,
    };
    setUpdateImpact(null);
    mutations.create.mutate(
      config,
      {
        onSuccess: (saved) => navigate(`/pipelines/${saved.id}/edit`),
        onError: (error) => setLocalError(error.message),
      },
    );
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
        {pipelineKind === "query" && <ExecutionSettings catalog={catalog.data} config={draft} onChange={setDraft} />}
      </section>

      {(localError || saveError) && <p className="form-error pipeline-editor-error" role="alert">{localError ?? saveError?.message}</p>}
      {issues.length > 0 && <div className="validation-summary" role="alert"><strong>Configuration needs attention</strong><ul>{issues.map((issue) => <li key={issue}>{issue}</li>)}</ul></div>}

      <div className="pipeline-editor-layout">
        <section className="pipeline-graph" aria-label="Pipeline graph">
          {pipelineKind !== "query" && <GraphSection label="Representation / indexing" stages={REPRESENTATION_STAGES} selected={selectedStage} config={draft} onSelect={setSelectedStage} />}
          {pipelineKind !== "ingestion" && <BranchingGraph
            config={draft}
            onAddBranch={(stage, branchId) => setDraft((current) => current ? addQueryBranch(current, stage, branchId) : current)}
            onRemoveBranch={(branchId) => setDraft((current) => current ? removeQueryBranch(current, branchId) : current)}
            onSelect={setSelectedStage}
            selected={selectedStage}
          />}
          {representationChanged && <p className="compatibility-warning"><strong>Representation changed.</strong> Changing parser, chunking, embedding, or vector settings may require a compatible index before this pipeline can execute.</p>}
        </section>
        <StageInspector
          catalog={catalog.data}
          definition={selectedDefinition}
          models={models.data}
          onChange={(value) => setDraft((current) => current ? updateStage(current, selectedDefinition, value) : current)}
          onAddAssessor={() => setDraft((current) => current ? updateAssessors(current, [...readAssessors(current), defaultStageValue(stageByKey(QUERY_STAGES, "assessors"), catalog.data, models.data)]) : current)}
          onRemoveAssessor={(index) => setDraft((current) => current ? updateAssessors(current, readAssessors(current).filter((_, itemIndex) => itemIndex !== index)) : current)}
          onChangeAssessor={(index, value) => setDraft((current) => current ? updateAssessors(current, readAssessors(current).map((item, itemIndex) => itemIndex === index ? value : item)) : current)}
          value={selectedValue}
        />
      </div>
      {updateImpact && (
        <PipelineUpdateDialog
          impact={updateImpact}
          pending={mutations.create.isPending || mutations.update.isPending}
          error={saveError}
          onClose={() => setUpdateImpact(null)}
          onCreateCopy={createCopy}
          onUpdateExisting={updateExisting}
        />
      )}
    </main>
  );
}

function PipelineUpdateDialog({
  impact,
  pending,
  error,
  onClose,
  onCreateCopy,
  onUpdateExisting,
}: {
  impact: PipelineUpdateImpact;
  pending: boolean;
  error: Error | null;
  onClose: () => void;
  onCreateCopy: () => void;
  onUpdateExisting: () => void;
}) {
  return (
    <Dialog title="Pipeline is in use" onClose={onClose}>
      <div className="impact-dialog">
        <p>
          This pipeline is selected by {impact.conversation_count} conversation
          {impact.conversation_count === 1 ? "" : "s"} and {impact.project_count ?? 0} project
          {(impact.project_count ?? 0) === 1 ? "" : "s"}. Choose how to apply the draft safely.
        </p>
        {impact.compatibility_reason && (
          <p className="form-error" role="alert">{impact.compatibility_reason}</p>
        )}
        <div className="impact-changes">
          <strong>Changed fields</strong>
          {impact.changes.length === 0 ? (
            <span>No configuration fields changed.</span>
          ) : (
            <ul>
              {impact.changes.map((change) => (
                <li key={change.path}>
                  <code>{formatChangePath(change.path)}</code>
                  {change.representation_affecting && <small> representation-affecting</small>}
                </li>
              ))}
            </ul>
          )}
        </div>
        <button className="impact-choice impact-choice-default" disabled={pending} onClick={onCreateCopy} type="button">
          <strong>Create a copy (recommended)</strong>
          <span>Keep existing conversations unchanged and save the draft as a new pipeline.</span>
        </button>
        <button className="impact-choice" disabled={pending || !impact.can_migrate} onClick={onUpdateExisting} type="button">
          <strong>Update existing pipeline</strong>
          <span>{impact.can_migrate ? "Migrate all conversations to the new version atomically." : "Unavailable until the representation/index is compatible."}</span>
        </button>
        {error && <p className="form-error" role="alert">{error.message}</p>}
        <footer><Button variant="ghost" onClick={onClose}>Cancel</Button></footer>
      </div>
    </Dialog>
  );
}

function formatChangePath(path: string): string {
  return path.replaceAll(".", " → ");
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

function BranchingGraph({
  config,
  selected,
  onSelect,
  onAddBranch,
  onRemoveBranch,
}: {
  config: PipelineConfig;
  selected: StageKey;
  onSelect: (stage: StageKey) => void;
  onAddBranch: (stage: BranchStart, branchId?: string) => void;
  onRemoveBranch: (branchId: string) => void;
}) {
  const layout = useMemo(() => buildBranchGraphLayout(config), [config]);
  return (
    <section className="graph-section branching-graph-section">
      <div className="graph-section-heading">
        <div>
          <h2>Query processing</h2>
          <p>Branch retrieval, reranking, or assessment while keeping shared stages centered.</p>
        </div>
        {readExecutionStrategy(config) === "branching" && <span className="graph-status">{layout.branches.length} branches</span>}
      </div>
      <div className="graph-canvas-scroll">
        <div className="graph-canvas" style={{ height: layout.height, minWidth: layout.width }}>
          <svg aria-hidden="true" className="graph-connectors" height={layout.height} viewBox={`0 0 ${layout.width} ${layout.height}`} width={layout.width}>
            {layout.edges.map((edge) => {
              const from = layout.nodes.find((node) => node.key === edge.from);
              const to = layout.nodes.find((node) => node.key === edge.to);
              if (!from || !to) return null;
              const startX = from.x + GRAPH_NODE_WIDTH;
              const startY = from.y + GRAPH_NODE_HEIGHT / 2;
              const endX = to.x;
              const endY = to.y + GRAPH_NODE_HEIGHT / 2;
              const midpoint = startX + (endX - startX) / 2;
              return <path className="graph-connector" d={`M ${startX} ${startY} H ${midpoint} V ${endY} H ${endX}`} key={`${edge.from}-${edge.to}`} />;
            })}
          </svg>
          {layout.branches.map((branch) => (
            <div className="graph-branch-controls" key={branch.id} style={{ top: branch.center - 24 }}>
              <span className="graph-branch-label">{branch.label}</span>
              {branch.removable && <button aria-label={`Delete ${branch.label}`} className="graph-delete-branch" onClick={() => onRemoveBranch(branch.id)} title={`Delete ${branch.label}`} type="button">×</button>}
            </div>
          ))}
          {layout.nodes.map((node) => {
            const configured = Boolean(readStage(config, node.definition));
            const branch = node.branchId ? layout.branches.find((item) => item.id === node.branchId) : undefined;
            const caption = node.shared ? "Shared" : branch?.label;
            const branchStart = stageToBranchStart(node.definition.key);
            return (
              <div className={`graph-node-position ${node.shared ? "shared-node" : "branch-node"}`} key={node.key} style={{ left: node.x, top: node.y }}>
                <StageNode
                  caption={caption}
                  configured={configured}
                  definition={node.definition}
                  selected={selected === node.definition.key}
                  onSelect={() => onSelect(node.definition.key)}
                />
                {branchStart && <button aria-label={`Add branch after ${node.definition.label}`} className="graph-add-branch" onClick={() => onAddBranch(branchStart, node.branchId)} title={`Add branch at ${node.definition.label}`} type="button">+</button>}
              </div>
            );
          })}
        </div>
      </div>
      <p className="graph-help">Use <strong>+</strong> below Retriever, Reranker, or Assessors to split the flow. Shared stages are reused automatically.</p>
    </section>
  );
}

function StageNode({ definition, selected, configured, caption, onSelect }: { definition: StageDefinition; selected: boolean; configured: boolean; caption?: string; onSelect: () => void }) {
  return <button className={`graph-node ${selected ? "selected" : ""} ${configured ? "configured" : "missing"}`} onClick={onSelect} type="button"><span>{definition.label}</span>{caption ? <small className="graph-node-caption">{caption}</small> : !configured && <small>Configure</small>}</button>;
}

/** Build a centered, column-based DAG layout for query branches. */
export function buildBranchGraphLayout(config: PipelineConfig): BranchGraphLayout {
  const branching = readExecutionStrategy(config) === "branching";
  const branches = queryBranches(config, branching);
  const rowCenters = branches.map((_, index) => GRAPH_TOP_PADDING + GRAPH_NODE_HEIGHT / 2 + index * GRAPH_ROW_GAP);
  const stageColumns = QUERY_STAGES.map((stage, index) => ({ stage, index }));
  const nodeMap = new Map<string, GraphNodeLayout>();
  const edgeMap = new Map<string, GraphEdgeLayout>();
  const columnStep = GRAPH_NODE_WIDTH + GRAPH_COLUMN_GAP;
  const xFor = (index: number): number => 22 + index * columnStep;
  const stageIndex = (stage: StageKey): number => QUERY_STAGES.findIndex((item) => item.key === stage);
  const sharedY = (index: number): number => {
    const descendants = branches.flatMap((branch, branchIndex) => {
      const startIndex = branching ? stageIndex(branchStartToStage(branch.start_at)) : 0;
      return index < startIndex ? [rowCenters[branchIndex]] : [];
    });
    return average(descendants.length > 0 ? descendants : rowCenters);
  };
  const nodeKeyFor = (stage: StageDefinition, branch: QueryBranch): { key: string; shared: boolean } => {
    if (!branching || stage.section !== "query" || stage.key === "decision_policy" || stage.key === "reasoner" || stage.key === "renderer") {
      return { key: `shared-${stage.key}`, shared: true };
    }
    const startIndex = stageIndex(branchStartToStage(branch.start_at));
    const currentIndex = stageIndex(stage.key);
    return currentIndex < startIndex
      ? { key: `shared-${stage.key}`, shared: true }
      : { key: `${branch.id}-${stage.key}`, shared: false };
  };
  const ensureNode = (stage: StageDefinition, branch: QueryBranch): GraphNodeLayout => {
    const identity = nodeKeyFor(stage, branch);
    const existing = nodeMap.get(identity.key);
    if (existing) return existing;
    const node: GraphNodeLayout = {
      key: identity.key,
      definition: stage,
      x: xFor(stageColumns.find((item) => item.stage.key === stage.key)?.index ?? 0),
      y: (identity.shared ? sharedY(stageIndex(stage.key)) : rowCenters[branches.findIndex((item) => item.id === branch.id)]) - GRAPH_NODE_HEIGHT / 2,
      shared: identity.shared,
      branchId: identity.shared ? undefined : branch.id,
    };
    nodeMap.set(node.key, node);
    return node;
  };
  const connect = (from: GraphNodeLayout, to: GraphNodeLayout): void => {
    if (from.key === to.key) return;
    const key = `${from.key}-${to.key}`;
    if (!edgeMap.has(key)) edgeMap.set(key, { from: from.key, to: to.key });
  };

  for (const branch of branches) {
    let previous: GraphNodeLayout | undefined;
    for (const { stage } of stageColumns) {
      const current = ensureNode(stage, branch);
      if (previous) connect(previous, current);
      previous = current;
    }
  }

  const width = xFor(QUERY_STAGES.length - 1) + GRAPH_NODE_WIDTH + 22;
  const height = GRAPH_TOP_PADDING + Math.max(1, branches.length - 1) * GRAPH_ROW_GAP + GRAPH_NODE_HEIGHT + GRAPH_BOTTOM_PADDING;
  return {
    nodes: [...nodeMap.values()],
    edges: [...edgeMap.values()],
    branches: branches.map((branch, index) => ({
      id: branch.id,
      label: branch.id === "pipeline" ? "Pipeline" : branch.id,
      center: rowCenters[index],
      removable: branching,
    })),
    width,
    height,
  };
}

function queryBranches(config: PipelineConfig, branching: boolean): QueryBranch[] {
  if (!branching) return [{ id: "pipeline", start_at: "retriever" }];
  const execution = isRecord(config.query?.execution) ? config.query.execution : {};
  const branches = Array.isArray(execution.branches)
    ? execution.branches.filter(isRecord).map((branch, index) => normalizeBranch(branch, index))
    : [];
  return branches.length > 0 ? branches : [{ id: "branch-1", start_at: "retriever" }, { id: "branch-2", start_at: "retriever" }];
}

function normalizeBranch(branch: Record<string, unknown>, index: number): QueryBranch {
  const startAt = branch.start_at === "reranker" || branch.start_at === "assessor" ? branch.start_at : "retriever";
  return {
    id: typeof branch.id === "string" && branch.id.trim() ? branch.id : `branch-${index + 1}`,
    start_at: startAt,
    ...(typeof branch.retriever === "string" ? { retriever: branch.retriever } : {}),
    ...(typeof branch.reranker === "string" ? { reranker: branch.reranker } : {}),
    ...(Array.isArray(branch.assessors) ? { assessors: branch.assessors.filter((item): item is string => typeof item === "string") } : {}),
  };
}

function branchStartToStage(start: BranchStart): StageKey {
  return start === "assessor" ? "assessors" : start;
}

function stageToBranchStart(stage: StageKey): BranchStart | undefined {
  if (stage === "assessors") return "assessor";
  return stage === "retriever" || stage === "reranker" ? stage : undefined;
}

function average(values: number[]): number {
  return values.reduce((total, value) => total + value, 0) / values.length;
}

/** Add a branch at a supported stage while retaining the existing config shape. */
export function addQueryBranch(config: PipelineConfig, startAt: BranchStart, afterBranchId?: string): PipelineConfig {
  const execution = isRecord(config.query?.execution) ? { ...config.query.execution } : {};
  const existing = Array.isArray(execution.branches) ? execution.branches.filter(isRecord).map((branch, index) => normalizeBranch(branch, index)) : [];
  const branchId = nextBranchId(existing.length < 2 ? [...existing, { id: "branch-1", start_at: startAt }] : existing);
  const newBranch: QueryBranch = { id: branchId, start_at: startAt };
  const insertionIndex = afterBranchId
    ? Math.max(0, existing.findIndex((branch) => branch.id === afterBranchId) + 1)
    : existing.length;
  const branches = existing.length < 2
    ? [{ ...(existing[0] ?? { id: "branch-1" }), start_at: startAt }, newBranch]
    : [...existing.slice(0, insertionIndex), newBranch, ...existing.slice(insertionIndex)];
  return {
    ...config,
    query: {
      ...(config.query ?? {}),
      execution: { ...execution, strategy: "branching", branches },
    },
  };
}

/** Remove a branch; a single remaining branch falls back to linear execution. */
export function removeQueryBranch(config: PipelineConfig, branchId: string): PipelineConfig {
  const execution = isRecord(config.query?.execution) ? { ...config.query.execution } : {};
  const branches = Array.isArray(execution.branches) ? execution.branches.filter(isRecord).map((branch, index) => normalizeBranch(branch, index)).filter((branch) => branch.id !== branchId) : [];
  if (branches.length === 0) return config;
  return {
    ...config,
    query: {
      ...(config.query ?? {}),
      execution: { ...execution, strategy: branches.length > 1 ? "branching" : "linear", branches },
    },
  };
}

function nextBranchId(branches: QueryBranch[]): string {
  const used = new Set(branches.map((branch) => branch.id));
  let index = branches.length + 1;
  while (used.has(`branch-${index}`)) index += 1;
  return `branch-${index}`;
}

function StageInspector({ catalog, definition, models, value, onChange, onAddAssessor, onRemoveAssessor, onChangeAssessor }: { catalog: PipelineCatalog; definition: StageDefinition; models: ModelCatalog; value: Record<string, unknown> | undefined; onChange: (value: Record<string, unknown>) => void; onAddAssessor: () => void; onRemoveAssessor: (index: number) => void; onChangeAssessor: (index: number, value: Record<string, unknown>) => void }) {
  const stage = catalog.stages[definition.catalogKey];
  const assessorValues = definition.key === "assessors" ? readAssessorValue(value) : [];
  const retrieverCapabilities = isRecord(catalog.compatibility.retriever_capabilities)
    ? catalog.compatibility.retriever_capabilities
    : undefined;
  const selectedRetriever = typeof value?.type === "string" ? value.type : undefined;
  const requirements = definition.key === "retriever" && selectedRetriever && retrieverCapabilities
    ? retrieverCapabilities[selectedRetriever]
    : undefined;
  const requirementNames = Array.isArray(requirements)
    ? requirements.filter((item): item is string => typeof item === "string")
    : [];
  return (
    <aside className="pipeline-inspector">
      <div className="inspector-heading"><span>{definition.section === "representation" ? "Representation / indexing" : "Query processing"}</span><h2>{definition.label}</h2><p>{definition.description}</p></div>
      {!stage ? <p className="form-error">This stage is not available in the backend catalog.</p> : definition.key === "assessors" ? (
        <div className="assessor-editor">{assessorValues.map((assessor, index) => <div className="assessor-item" key={index}><div className="assessor-item-heading"><strong>Assessor {index + 1}</strong>{assessorValues.length > 1 && <button onClick={() => onRemoveAssessor(index)} type="button">Remove</button>}</div><SchemaFields stage={stage} value={assessor} models={models} stageKey={definition.key} onChange={(next) => onChangeAssessor(index, next)} /></div>)}<Button onClick={onAddAssessor}>＋ Add assessor</Button></div>
      ) : (
        <SchemaFields stage={stage} value={value ?? {}} models={models} stageKey={definition.key} onChange={onChange} />
      )}
      {definition.key === "embedding" && <p className="inspector-warning">Embedding changes alter the representation identity and may require a new compatible index.</p>}
      {requirementNames.length > 0 && <p className="inspector-warning">This retriever requires: {requirementNames.join(", ")}. The project index must provide these capabilities.</p>}
      <p className="inspector-catalog-note">Options and parameter constraints are provided by the backend catalog.</p>
    </aside>
  );
}

function SchemaFields({ stage, value, models, stageKey, onChange }: { stage: CatalogStage; value: Record<string, unknown>; models: ModelCatalog; stageKey: StageKey; onChange: (value: Record<string, unknown>) => void }) {
  const properties = stage.schema.properties ?? {};
  const required = new Set(stage.schema.required ?? []);
  return <div className="schema-fields">{Object.entries(properties).map(([field, fieldSchema]) => <SchemaField key={field} field={field} schema={fieldSchema} required={required.has(field)} value={value[field]} implementations={field === "type" ? stage.implementations : undefined} modelOptions={field === "model_ref" ? (stageKey === "embedding" ? Object.keys(models.embedding) : stageKey === "reasoner" ? Object.keys(models.reasoning) : undefined) : undefined} onChange={(next) => { let updated = { ...value }; if (field === "type" && next !== value.type) { const allowed = new Set(Object.keys(properties)); updated = Object.fromEntries(Object.entries(updated).filter(([key]) => key === "type" || allowed.has(key))); } if (next === undefined) delete updated[field]; else updated[field] = next; onChange(updated); }} />)}</div>;
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
  if (!section) return undefined;
  const value = section[definition.key];
  if (definition.key === "assessors") return readAssessorValue(value)[0];
  return isRecord(value) ? value : undefined;
}

function readAssessors(config: PipelineConfig): Record<string, unknown>[] {
  return readAssessorValue(config.query?.assessors);
}

function readAssessorValue(value: unknown): Record<string, unknown>[] {
  if (isRecord(value)) return [value];
  if (!Array.isArray(value)) return [];
  return value.filter(isRecord);
}

function updateStage(config: PipelineConfig, definition: StageDefinition, value: Record<string, unknown>): PipelineConfig {
  const next = cloneConfig(config);
  if (definition.key === "assessors") next.query = { ...(next.query ?? {}), assessors: [value] };
  else next[definition.section] = { ...(next[definition.section] ?? {}), [definition.key]: value };
  return next;
}

function updateAssessors(config: PipelineConfig, values: Record<string, unknown>[]): PipelineConfig {
  const next = cloneConfig(config);
  next.query = { ...(next.query ?? {}), assessors: values };
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

function validateDraft(config: PipelineConfig | null, name: string, version: string, stages: StageDefinition[], catalog?: PipelineCatalog, models?: ModelCatalog): string[] {
  if (!config || !catalog || !models) return ["Pipeline configuration is still loading."];
  const issues: string[] = [];
  if (!name.trim()) issues.push("Pipeline name is required.");
  if (!/^\d+$/.test(version) || Number(version) < 1) issues.push("Version must be a positive integer.");
  if (config.kind === "query" && readExecutionStrategy(config) === "branching") {
    const execution = isRecord(config.query?.execution) ? config.query.execution : {};
    const branches = Array.isArray(execution.branches) ? execution.branches.filter(isRecord) : [];
    const ids = branches.map((branch) => branch.id).filter((id): id is string => typeof id === "string");
    if (branches.length < 2) issues.push("Branching execution requires at least two branches.");
    if (new Set(ids).size !== ids.length) issues.push("Execution branch IDs must be unique.");
  }
  for (const definition of stages) {
    const value = readStage(config, definition);
    if (!value) issues.push(`${definition.label} must be configured.`);
    const stage = catalog.stages[definition.catalogKey];
    if (value && stage) {
      for (const field of stage.schema.required ?? []) {
        if (value[field] === undefined || value[field] === null || value[field] === "") {
          issues.push(`${definition.label} requires ${humanize(field)}.`);
        }
      }
      const type = value.type;
      if (typeof type === "string" && stage.implementations.length > 0 && !stage.implementations.includes(type)) issues.push(`${definition.label} uses an unsupported implementation.`);
      if (definition.key === "embedding" && typeof value.model_ref === "string" && !models.embedding[value.model_ref]) issues.push("Embedding model reference is unavailable.");
      if (definition.key === "reasoner" && typeof value.model_ref === "string" && !models.reasoning[value.model_ref]) issues.push("Reasoning model reference is unavailable.");
    }
  }
  return [...new Set(issues)];
}

function emptyConfig(kind: PipelineKind): PipelineConfig {
  return {
    schema_version: 1,
    kind,
    name: "new_pipeline",
    version: 1,
    description: "",
    is_default: false,
    ...(kind !== "query" ? { representation: {} } : {}),
    ...(kind !== "ingestion" ? { query: {} } : {}),
  };
}

function parsePipelineKind(value: string | null): PipelineKind | undefined {
  return value === "ingestion" || value === "query" ? value : undefined;
}

function stagesForKind(kind: PipelineKind): StageDefinition[] {
  if (kind === "ingestion") return REPRESENTATION_STAGES;
  if (kind === "query") return QUERY_STAGES;
  return QUERY_STAGES;
}

function cloneConfig(config: PipelineConfig): PipelineConfig {
  return orderPipelineStages(JSON.parse(JSON.stringify(config)) as PipelineConfig);
}

/**
 * Preserve the canonical query-stage order when a configuration is cloned or serialized.
 * Unknown query settings remain available after the registered stages.
 */
export function orderPipelineStages(config: PipelineConfig): PipelineConfig {
  if (!config.query) return config;
  const orderedQuery: Record<string, unknown> = {};
  for (const stage of QUERY_STAGES) {
    if (Object.prototype.hasOwnProperty.call(config.query, stage.key)) {
      orderedQuery[stage.key] = config.query[stage.key];
    }
  }
  for (const [key, value] of Object.entries(config.query)) {
    if (!Object.prototype.hasOwnProperty.call(orderedQuery, key)) {
      orderedQuery[key] = value;
    }
  }
  return { ...config, query: orderedQuery };
}

function stageByKey(stages: StageDefinition[], key: StageKey): StageDefinition {
  const definition = stages.find((stage) => stage.key === key);
  if (!definition) throw new Error(`Pipeline stage is not registered: ${key}`);
  return definition;
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

function ExecutionSettings({ catalog, config, onChange }: { catalog: PipelineCatalog; config: PipelineConfig; onChange: (config: PipelineConfig) => void }) {
  const strategy = readExecutionStrategy(config);
  const execution = isRecord(config.query?.execution) ? config.query.execution : {};
  const branches = Array.isArray(execution.branches)
    ? execution.branches.filter(isRecord)
    : [];
  return (
    <div className="execution-settings">
      <label>Execution strategy<select value={strategy} onChange={(event) => onChange(updateExecutionStrategy(config, event.target.value as ExecutionStrategy))}>
        <option value="linear">Linear</option>
        <option value="branching">Branching</option>
        <option value="recursive">Recursive</option>
      </select></label>
      {strategy === "recursive" && <label>Maximum rounds<input min="2" max="100" type="number" value={typeof execution.max_rounds === "number" ? execution.max_rounds : 2} onChange={(event) => onChange(updateExecutionValue(config, "max_rounds", Math.max(2, Number(event.target.value) || 2)))} /></label>}
      {strategy === "branching" && <>
        <label>Branches<input value={branches.map((branch, index) => typeof branch.id === "string" ? branch.id : `branch-${index + 1}`).join(", ")} onChange={(event) => onChange(updateBranchIds(config, event.target.value))} /></label>
        <label>Evidence merge<select value={execution.evidence_merge === "union" || execution.evidence_merge === "rrf" ? execution.evidence_merge : "replace"} onChange={(event) => onChange(updateExecutionValue(config, "evidence_merge", event.target.value))}><option value="replace">Replace with selected branch</option><option value="union">Union branch evidence</option><option value="rrf">Reciprocal-rank fusion</option></select></label>
        <div className="execution-branches">{branches.map((branch, index) => <div className="execution-branch" key={typeof branch.id === "string" ? branch.id : index}>
          <strong>{typeof branch.id === "string" ? branch.id : `Branch ${index + 1}`}</strong>
          <label>Start at<select value={branch.start_at === "reranker" || branch.start_at === "assessor" ? branch.start_at : "retriever"} onChange={(event) => onChange(updateBranchField(config, index, "start_at", event.target.value))}><option value="retriever">Retriever</option><option value="reranker">Reranker</option><option value="assessor">Assessor</option></select></label>
          <label>Retriever<select value={typeof branch.retriever === "string" ? branch.retriever : ""} onChange={(event) => onChange(updateBranchField(config, index, "retriever", event.target.value))}><option value="">Use pipeline retriever</option>{catalog.stages.retriever?.implementations.map((implementation) => <option key={implementation} value={implementation}>{implementation}</option>)}</select></label>
          <label>Reranker<select value={typeof branch.reranker === "string" ? branch.reranker : ""} onChange={(event) => onChange(updateBranchField(config, index, "reranker", event.target.value))}><option value="">Use pipeline reranker</option>{catalog.stages.reranker?.implementations.map((implementation) => <option key={implementation} value={implementation}>{implementation}</option>)}</select></label>
          <label>Assessor<select value={Array.isArray(branch.assessors) && typeof branch.assessors[0] === "string" ? branch.assessors[0] : ""} onChange={(event) => onChange(updateBranchField(config, index, "assessor", event.target.value))}><option value="">Use pipeline assessors</option>{catalog.stages.assessor?.implementations.map((implementation) => <option key={implementation} value={implementation}>{implementation}</option>)}</select></label>
        </div>)}</div>
      </>}
    </div>
  );
}

export function readExecutionStrategy(config: PipelineConfig): ExecutionStrategy {
  const execution = config.query?.execution;
  if (isRecord(execution) && (execution.strategy === "branching" || execution.strategy === "recursive")) {
    return execution.strategy;
  }
  return "linear";
}

export function updateExecutionStrategy(config: PipelineConfig, strategy: ExecutionStrategy): PipelineConfig {
  const execution = isRecord(config.query?.execution) ? { ...config.query.execution } : {};
  if (strategy === "branching" && (!Array.isArray(execution.branches) || execution.branches.length < 2)) {
    const existing = Array.isArray(execution.branches) ? execution.branches.filter(isRecord) : [];
    execution.branches = [
      { ...(existing[0] ?? {}), id: typeof existing[0]?.id === "string" ? existing[0].id : "branch-1" },
      { ...(existing[1] ?? {}), id: typeof existing[1]?.id === "string" ? existing[1].id : "branch-2" },
    ];
  }
  if (strategy === "recursive" && typeof execution.max_rounds !== "number") {
    execution.max_rounds = 2;
  }
  execution.strategy = strategy;
  return { ...config, query: { ...(config.query ?? {}), execution } };
}

function updateExecutionValue(config: PipelineConfig, key: string, value: unknown): PipelineConfig {
  const execution = isRecord(config.query?.execution) ? { ...config.query.execution } : {};
  execution[key] = value;
  return { ...config, query: { ...(config.query ?? {}), execution } };
}

function updateBranchIds(config: PipelineConfig, value: string): PipelineConfig {
  const ids = value.split(",").map((item) => item.trim()).filter(Boolean);
  const execution = isRecord(config.query?.execution) ? { ...config.query.execution } : {};
  const previous = Array.isArray(execution.branches) ? execution.branches.filter(isRecord) : [];
  execution.branches = ids.map((id, index) => ({
    ...(previous[index] ?? {}),
    id,
  }));
  return { ...config, query: { ...(config.query ?? {}), execution } };
}

function updateBranchField(config: PipelineConfig, index: number, key: "start_at" | "retriever" | "reranker" | "assessor", value: string): PipelineConfig {
  const execution = isRecord(config.query?.execution) ? { ...config.query.execution } : {};
  const branches = Array.isArray(execution.branches) ? execution.branches.filter(isRecord).map((branch) => ({ ...branch })) : [];
  if (!branches[index]) return config;
  if (key === "assessor") {
    if (value) branches[index].assessors = [value];
    else delete branches[index].assessors;
  } else if (value) {
    branches[index][key] = value;
  } else {
    delete branches[index][key];
  }
  execution.branches = branches;
  return { ...config, query: { ...(config.query ?? {}), execution } };
}
