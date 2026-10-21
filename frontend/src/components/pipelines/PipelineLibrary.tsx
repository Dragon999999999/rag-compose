import { useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";

import type { Pipeline, PipelineConfig } from "../../api/types";
import { Dialog } from "../common/Dialog";
import { Button, EmptyState, ErrorState, StatusBadge } from "../common/ui";
import { usePipeline, usePipelineMutations, usePipelines } from "../../features/hooks";
import "./PipelinePages.css";

type StatusFilter = "all" | "active" | "inactive";
type SortOrder = "updated" | "name" | "version";

/** Provide a central, searchable inventory of persisted RAG pipelines. */
export function PipelineLibrary() {
  const navigate = useNavigate();
  const pipelines = usePipelines(true);
  const mutations = usePipelineMutations();
  const [search, setSearch] = useState("");
  const [status, setStatus] = useState<StatusFilter>("all");
  const [sort, setSort] = useState<SortOrder>("updated");
  const [showCreate, setShowCreate] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<Pipeline | null>(null);

  const visiblePipelines = useMemo(() => {
    const query = search.trim().toLowerCase();
    return [...(pipelines.data ?? [])]
      .filter((pipeline) => {
        const matchesSearch =
          !query ||
          [pipeline.name, pipeline.description, pipeline.source_kind]
            .filter(Boolean)
            .some((value) => value!.toLowerCase().includes(query));
        const matchesStatus =
          status === "all" || (status === "active" ? pipeline.active : !pipeline.active);
        return matchesSearch && matchesStatus;
      })
      .sort((left, right) => {
        if (sort === "name") return left.name.localeCompare(right.name);
        if (sort === "version") return right.version - left.version;
        return right.updated_at.localeCompare(left.updated_at);
      });
  }, [pipelines.data, search, sort, status]);

  if (pipelines.isLoading) {
    return <main className="pipeline-page"><div className="pipeline-loading">Loading pipelines…</div></main>;
  }
  if (pipelines.error) {
    return <main className="pipeline-page"><ErrorState error={pipelines.error} retry={() => pipelines.refetch()} /></main>;
  }

  const confirmDelete = (): void => {
    if (!deleteTarget) return;
    mutations.deactivate.mutate(deleteTarget.id, {
      onSuccess: () => setDeleteTarget(null),
    });
  };

  return (
    <main className="pipeline-page">
      <header className="pipeline-page-header">
        <div>
          <Link className="back-link" to="/projects">← Projects</Link>
          <h1>Pipeline library</h1>
          <p>Discover and manage the configurations that power grounded answers.</p>
        </div>
        <Button variant="primary" onClick={() => setShowCreate(true)}>＋ Create pipeline</Button>
      </header>

      <section className="pipeline-toolbar" aria-label="Pipeline filters">
        <label className="pipeline-search">
          <span className="sr-only">Search pipelines</span>
          <input
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Search pipelines…"
            value={search}
          />
        </label>
        <label>
          <span className="sr-only">Filter by status</span>
          <select aria-label="Filter by status" onChange={(event) => setStatus(event.target.value as StatusFilter)} value={status}>
            <option value="all">All statuses</option>
            <option value="active">Active</option>
            <option value="inactive">Inactive</option>
          </select>
        </label>
        <label>
          <span className="sr-only">Sort pipelines</span>
          <select aria-label="Sort pipelines" onChange={(event) => setSort(event.target.value as SortOrder)} value={sort}>
            <option value="updated">Recently modified</option>
            <option value="name">Name</option>
            <option value="version">Version</option>
          </select>
        </label>
      </section>

      {visiblePipelines.length === 0 ? (
        <EmptyState title={pipelines.data?.length ? "No matching pipelines" : "No pipelines yet"}>
          {pipelines.data?.length
            ? "Try a different search or status filter."
            : "Create a pipeline to make a RAG configuration available to conversations."}
        </EmptyState>
      ) : (
        <div className="pipeline-table-wrap">
          <table className="pipeline-table">
            <thead>
              <tr>
                <th scope="col">Pipeline</th>
                <th scope="col">Embedding</th>
                <th scope="col">Reasoning</th>
                <th scope="col">Version</th>
                <th scope="col">Status</th>
                <th scope="col">Conversations</th>
                <th scope="col">Modified</th>
                <th scope="col"><span className="sr-only">Actions</span></th>
              </tr>
            </thead>
            <tbody>
              {visiblePipelines.map((pipeline) => (
                <PipelineRow
                  key={pipeline.id}
                  pipeline={pipeline}
                  onDuplicate={() => navigate(`/pipelines/new?source=${pipeline.id}`)}
                  onDelete={() => setDeleteTarget(pipeline)}
                  onEdit={() => navigate(`/pipelines/${pipeline.id}/edit`)}
                  onSetDefault={() => mutations.setDefault.mutate(pipeline.id)}
                  settingDefault={mutations.setDefault.isPending && mutations.setDefault.variables === pipeline.id}
                />
              ))}
            </tbody>
          </table>
        </div>
      )}

      <p className="pipeline-page-note">
        Pipeline definitions are validated and persisted by the backend. Changes affect new executions; historical responses retain their recorded configuration hash.
      </p>

      {showCreate && <CreatePipelineDialog onClose={() => setShowCreate(false)} onChoose={(mode) => navigate(`/pipelines/new?mode=${mode}`)} />}
      {deleteTarget && (
        <Dialog title={`Deactivate ${deleteTarget.name}?`} onClose={() => setDeleteTarget(null)}>
          <p>This will remove the pipeline from active selectors. Existing conversations and historical executions remain traceable.</p>
          {mutations.deactivate.error && <p className="form-error" role="alert">{mutations.deactivate.error.message}</p>}
          <footer>
            <Button variant="ghost" onClick={() => setDeleteTarget(null)}>Cancel</Button>
            <Button variant="danger" disabled={mutations.deactivate.isPending} onClick={confirmDelete}>
              {mutations.deactivate.isPending ? "Deactivating…" : "Deactivate"}
            </Button>
          </footer>
        </Dialog>
      )}
    </main>
  );
}

function PipelineRow({
  pipeline,
  onDuplicate,
  onDelete,
  onEdit,
  onSetDefault,
  settingDefault,
}: {
  pipeline: Pipeline;
  onDuplicate: () => void;
  onDelete: () => void;
  onEdit: () => void;
  onSetDefault: () => void;
  settingDefault: boolean;
}) {
  const detail = usePipeline(pipeline.id);
  const embedding = modelReference(detail.data?.config, "representation");
  const reasoning = modelReference(detail.data?.config, "query");
  return (
    <tr>
      <th scope="row">
        <button className="pipeline-name" onClick={onEdit} type="button">{pipeline.name}</button>
        <span className="pipeline-description">{pipeline.description || "No description"}</span>
        <span className="pipeline-source">{pipeline.source_kind ?? "configuration"}</span>
      </th>
      <td>{embedding ?? "—"}</td>
      <td>{reasoning ?? "—"}</td>
      <td>v{pipeline.version}</td>
      <td>
        <div className="pipeline-statuses">
          <StatusBadge status={pipeline.active ? "Active" : "Inactive"} />
          {pipeline.is_default && <span className="default-badge">Default</span>}
        </div>
      </td>
      <td>{pipeline.conversation_count ?? "—"}</td>
      <td><time dateTime={pipeline.updated_at}>{formatDate(pipeline.updated_at)}</time></td>
      <td>
        <div className="pipeline-actions">
          <Button onClick={onEdit}>Edit</Button>
          <Button onClick={onDuplicate}>Duplicate</Button>
          {!pipeline.is_default && pipeline.active && <Button disabled={settingDefault} onClick={onSetDefault}>{settingDefault ? "Setting…" : "Set default"}</Button>}
          {pipeline.active && <Button variant="danger" onClick={onDelete}>Deactivate</Button>}
        </div>
      </td>
    </tr>
  );
}

function CreatePipelineDialog({ onClose, onChoose }: { onClose: () => void; onChoose: (mode: "default" | "blank") => void }) {
  return (
    <Dialog title="Create pipeline" onClose={onClose}>
      <p>Choose a starting point. The editor sends the final typed configuration to the backend for validation.</p>
      <div className="create-pipeline-options">
        <button onClick={() => onChoose("default")} type="button"><strong>From default</strong><span>Start with the current default RAG configuration.</span></button>
        <button onClick={() => onChoose("blank")} type="button"><strong>Blank configuration</strong><span>Build the complete specification using the registered stages.</span></button>
      </div>
      <footer><Button variant="ghost" onClick={onClose}>Cancel</Button></footer>
    </Dialog>
  );
}

function modelReference(config: PipelineConfig | undefined, section: "representation" | "query"): string | null {
  const sectionValue = config?.[section];
  if (!sectionValue || typeof sectionValue !== "object") return null;
  const modelSection = section === "representation"
    ? (sectionValue as Record<string, unknown>).embedding
    : (sectionValue as Record<string, unknown>).reasoner;
  if (!modelSection || typeof modelSection !== "object") return null;
  const modelRef = (modelSection as Record<string, unknown>).model_ref;
  return typeof modelRef === "string" ? modelRef : null;
}

function formatDate(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "Unknown" : new Intl.DateTimeFormat(undefined, { dateStyle: "medium" }).format(date);
}
