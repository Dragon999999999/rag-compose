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
  const defaultCompatibility = queryCompatibility.find(
    (item) => item.pipeline.is_default,
  ) ?? queryCompatibility.find((item) => item.pipeline.name === "query_basic_v1");
  const selectedPipelineId = chat?.pipeline_id ?? defaultCompatibility?.pipeline.id;
  const selectedCompatibility = queryCompatibility.find(
    (item) => item.pipeline.id === selectedPipelineId,
  );
  const incompatiblePipelines = queryCompatibility.filter(
    (item) => !item.compatible,
  );
  const indexReady = Boolean(project?.index_capabilities?.capabilities?.length);

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
                <option value="">
                  Application default{defaultCompatibility && (defaultCompatibility.compatible ? " · ✓" : " · ⚠")}
                </option>
              )}
              {chat.pipeline_id &&
                !queryCompatibility.some((item) => item.pipeline.id === chat.pipeline_id) && (
                  <option disabled value={chat.pipeline_id}>
                    {chat.pipeline_id} (unavailable)
                  </option>
                )}
              {queryCompatibility.filter((item) => item.compatible).map((item) => (
                <option key={item.pipeline.id} value={item.pipeline.id}>✓ {item.pipeline.name} (v{item.pipeline.version})</option>
              ))}
              {queryCompatibility.filter((item) => !item.compatible).map((item) => (
                <option disabled key={item.pipeline.id} value={item.pipeline.id}>⚠ {item.pipeline.name} — incompatible</option>
              ))}
            </select>
            {incompatiblePipelines.length > 0 && (
              <details className="pipeline-compatibility-details">
                <summary>Inspect compatibility</summary>
                <div className="pipeline-compatibility-options">
                  {incompatiblePipelines.map((item) => (
                    <details key={item.pipeline.id}>
                      <summary>
                        <span aria-hidden="true" className="pipeline-status-indicator incompatible">⚠</span>
                        {item.pipeline.name}
                      </summary>
                      <p>
                        Missing capabilities: {item.missing_capabilities.length > 0
                          ? item.missing_capabilities.join(", ")
                          : "not reported"}.
                      </p>
                      {item.reason && <p>{item.reason}</p>}
                    </details>
                  ))}
                </div>
              </details>
            )}
          </label>
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
      {project && <p className="index-status">Index: {indexReady ? "Ready" : "Not indexed"}{project.ingestion_pipeline_id ? ` · ${project.ingestion_pipeline_id}` : ""}</p>}
      {(pipelineError || compatibilityError || ingestionError) && <p className="pipeline-error" role="alert">{pipelineError?.message ?? compatibilityError?.message ?? ingestionError?.message}</p>}
      {chat && selectedCompatibility && !selectedCompatibility.compatible && (
        <div className="active-pipeline-compatibility" role="alert">
          <strong>Selected query pipeline is incompatible with this index.</strong>
          <span>
            {indexReady ? "The index is ready, but it does not provide what this pipeline requires." : "The project is not indexed yet, so this pipeline cannot be verified as compatible."}
          </span>
          <span>Missing capabilities: {selectedCompatibility.missing_capabilities.length > 0 ? selectedCompatibility.missing_capabilities.join(", ") : "not reported"}.</span>
        </div>
      )}
    </header>
  );
}
