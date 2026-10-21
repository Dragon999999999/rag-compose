import type { Chat, Pipeline, Project } from "../../api/types";
import { Button } from "../common/ui";

type ProjectHeaderProps = {
  project?: Project;
  chat?: Chat;
  onWorkspaceToggle: () => void;
  onNewChat: () => void;
  hasChats: boolean;
  view: "chat" | "sources";
  pipelines: Pipeline[];
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
  pipelines,
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
                !pipelines.some((pipeline) => pipeline.id === chat.pipeline_id) && (
                  <option disabled value={chat.pipeline_id}>
                    {chat.pipeline_id} (unavailable)
                  </option>
                )}
              {pipelines.map((pipeline) => (
                <option key={pipeline.id} value={pipeline.id}>
                  {pipeline.name} (v{pipeline.version})
                </option>
              ))}
            </select>
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
      {pipelineError && <p className="pipeline-error" role="alert">{pipelineError.message}</p>}
    </header>
  );
}
