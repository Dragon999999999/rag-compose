import type { Chat, Pipeline, PipelineCompatibility, Project } from "../../api/types";
import { Button } from "../common/ui";

type ProjectHeaderProps = {
  project?: Project;
  chat?: Chat;
  onWorkspaceToggle: () => void;
  onNewChat: () => void;
  hasChats: boolean;
  view: "chat" | "sources";
  ingestionPipelines: Pipeline[];
  queryCompatibility: PipelineCompatibility[];
  compatibilityError?: Error | null;
  ingestionError?: Error | null;
  ingestionChanging?: boolean;
  onIngestionChange?: (pipelineId: string) => void;
  pipelineError?: Error | null;
  pipelineChanging?: boolean;
  onPipelineChange: (pipelineId: string) => void;
};

/** Show workspace navigation and actions that do not edit project metadata. */
export function ProjectHeader({
  project,
  chat,
  onWorkspaceToggle,
  onNewChat,
  hasChats,
  view,
  ingestionPipelines,
  queryCompatibility,
  compatibilityError,
  ingestionError,
  ingestionChanging,
  onIngestionChange,
  pipelineError,
  pipelineChanging,
  onPipelineChange,
}: ProjectHeaderProps) {
  return (
    <header className="project-header">
      <div className="header-identity">
        <div>
          <h1>{project?.name ?? "LLM Grounding"}</h1>
          <p>{chat?.name ?? "Select or create a chat"}</p>
        </div>
      </div>
      <div className="header-actions">
        {project && (
          <label className="pipeline-selector">
            <span>Indexing</span>
            <select
              aria-label="Project ingestion pipeline"
              disabled={ingestionChanging}
              onChange={(event) => {
                const nextPipelineId = event.target.value;
                const hasIndex = Boolean(project.index_capabilities?.capabilities?.length);
                if (hasIndex && nextPipelineId !== project.ingestion_pipeline_id) {
                  const confirmed = window.confirm("Changing the ingestion pipeline may require re-indexing. Continue? The backend will reject incompatible changes and no re-indexing will start automatically.");
                  if (!confirmed) return;
                }
                onIngestionChange?.(nextPipelineId);
              }}
              value={project.ingestion_pipeline_id ?? ""}
            >
              <option value="">Application default</option>
              {ingestionPipelines.map((pipeline) => <option key={pipeline.id} value={pipeline.id}>{pipeline.name} (v{pipeline.version})</option>)}
            </select>
          </label>
        )}
        {chat && (
          <label className="pipeline-selector">
            <span>Pipeline</span>
            <select
              aria-label="Conversation pipeline"
              disabled={pipelineChanging}
              onChange={(event) => onPipelineChange(event.target.value)}
              value={chat.pipeline_id ?? ""}
            >
              {!chat.pipeline_id && (
                <option value="">Application default</option>
              )}
              {chat.pipeline_id &&
                !queryCompatibility.some((item) => item.pipeline.id === chat.pipeline_id) && (
                  <option disabled value={chat.pipeline_id}>
                    {chat.pipeline_id} (unavailable)
                  </option>
                )}
              {queryCompatibility.filter((item) => item.compatible).map((item) => (
                <option key={item.pipeline.id} value={item.pipeline.id}>{item.pipeline.name} (v{item.pipeline.version})</option>
              ))}
              {queryCompatibility.filter((item) => !item.compatible).map((item) => (
                <option disabled key={item.pipeline.id} value={item.pipeline.id}>{item.pipeline.name} — unavailable</option>
              ))}
            </select>
          </label>
        )}
        {chat && queryCompatibility.some((item) => !item.compatible) && (
          <div className="pipeline-compatibility-list" aria-label="Unavailable query pipelines">
            {queryCompatibility.filter((item) => !item.compatible).map((item) => (
              <span key={item.pipeline.id}>Unavailable: {item.pipeline.name} — {item.reason ?? item.missing_capabilities.join(", ")}</span>
            ))}
          </div>
        )}
        <Button
          onClick={onWorkspaceToggle}
          disabled={!project || (view === "sources" && !hasChats)}
        >
          {view === "sources" ? "Chats" : "Sources"}
        </Button>
        <Button onClick={onNewChat} disabled={!project}>
          New chat
        </Button>
      </div>
      {project && <p className="index-status">Index: {project.index_capabilities?.capabilities?.length ? "Ready" : "Not indexed"}{project.ingestion_pipeline_id ? ` · ${project.ingestion_pipeline_id}` : ""}</p>}
      {(pipelineError || compatibilityError || ingestionError) && <p className="pipeline-error" role="alert">{pipelineError?.message ?? compatibilityError?.message ?? ingestionError?.message}</p>}
      {chat && queryCompatibility.some((item) => item.pipeline.id === chat.pipeline_id && !item.compatible) && <p className="pipeline-error" role="alert">{queryCompatibility.find((item) => item.pipeline.id === chat.pipeline_id)?.reason}</p>}
    </header>
  );
}
