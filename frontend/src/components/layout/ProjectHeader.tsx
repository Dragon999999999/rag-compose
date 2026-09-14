import type { Chat, Project } from "../../api/types";
import { Button } from "../common/ui";

type ProjectHeaderProps = {
  project?: Project;
  chat?: Chat;
  onSources: () => void;
  onNewChat: () => void;
};

/** Show workspace navigation and actions that do not edit project metadata. */
export function ProjectHeader({
  project,
  chat,
  onSources,
  onNewChat,
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
        <Button onClick={onSources} disabled={!project}>
          Sources
        </Button>
        <Button onClick={onNewChat} disabled={!project}>
          New chat
        </Button>
      </div>
    </header>
  );
}
