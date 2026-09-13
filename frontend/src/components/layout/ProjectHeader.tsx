import type { Chat, Project } from "../../api/types";
import { Button, IconButton } from "../common/ui";

type ProjectHeaderProps = {
  project?: Project;
  chat?: Chat;
  sidebarOpen: boolean;
  evidenceOpen: boolean;
  onSidebar: () => void;
  onSources: () => void;
  onNewChat: () => void;
  onSummary: () => void;
  onEvidence: () => void;
};

/** Show workspace navigation and actions that do not edit project metadata. */
export function ProjectHeader({
  project,
  chat,
  sidebarOpen,
  evidenceOpen,
  onSidebar,
  onSources,
  onNewChat,
  onSummary,
  onEvidence,
}: ProjectHeaderProps) {
  return (
    <header className="project-header">
      <div className="header-identity">
        <IconButton
          aria-label={sidebarOpen ? "Hide navigation" : "Show navigation"}
          onClick={onSidebar}
        >
          ☰
        </IconButton>
        <div>
          <h1>{project?.name ?? "LLM Grounding"}</h1>
          <p>{chat?.name ?? "Select or create a chat"}</p>
        </div>
      </div>
      <div className="header-actions">
        <Button onClick={onSummary} disabled={!chat}>
          Summarize
        </Button>
        <Button onClick={onSources} disabled={!project}>
          Sources
        </Button>
        <Button onClick={onNewChat} disabled={!project}>
          New chat
        </Button>
        <Button onClick={onEvidence}>
          {evidenceOpen ? "Hide evidence" : "Evidence"}
        </Button>
      </div>
    </header>
  );
}
