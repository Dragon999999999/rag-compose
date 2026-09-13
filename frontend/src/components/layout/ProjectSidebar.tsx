import { NavLink } from "react-router-dom";

import type { Chat, Project } from "../../api/types";
import { Button, IconButton } from "../common/ui";
import "./ProjectSidebar.css";

type ProjectSidebarProps = {
  projects: Project[];
  selectedProjectId?: string;
  chats: Chat[];
  onNewProject: () => void;
  onNewChat: () => void;
};

export function ProjectSidebar({
  projects,
  selectedProjectId,
  chats,
  onNewProject,
  onNewChat,
}: ProjectSidebarProps) {
  return (
    <aside className="project-sidebar">
      <div className="sidebar-title">Projects</div>
      <nav aria-label="Projects">
        {projects.map((project) => {
          const isSelected = project.id === selectedProjectId;
          return (
            <section className="project-tree" key={project.id}>
              <div className={`project-row ${isSelected ? "selected" : ""}`}>
                <NavLink to={`/projects/${project.id}/sources`}>{project.name}</NavLink>
                {isSelected && (
                  <IconButton
                    aria-label={`Create a chat in ${project.name}`}
                    onClick={onNewChat}
                    title="New chat"
                  >
                    ＋
                  </IconButton>
                )}
              </div>
              {isSelected && chats.length > 0 && (
                <div className="chat-tree">
                  {chats.map((chat) => (
                    <NavLink
                      key={chat.id}
                      to={`/projects/${project.id}/chats/${chat.id}`}
                      className={({ isActive }) =>
                        `chat-row ${isActive ? "selected" : ""}`
                      }
                    >
                      {chat.name}
                    </NavLink>
                  ))}
                </div>
              )}
            </section>
          );
        })}
      </nav>
      <Button variant="primary" className="new-project" onClick={onNewProject}>
        ＋ New project
      </Button>
    </aside>
  );
}
