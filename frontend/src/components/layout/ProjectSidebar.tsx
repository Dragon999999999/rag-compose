import { useEffect, useState, type FormEvent, type MouseEvent } from "react";
import { NavLink } from "react-router-dom";

import type { Chat, Project } from "../../api/types";
import { sortChatsByActivity } from "../../features/chatNavigation";
import { Button, IconButton } from "../common/ui";
import "./ProjectSidebar.css";

type ContextTargetBase =
  | { kind: "project"; item: Project }
  | { kind: "chat"; item: Chat };

type ContextTarget = ContextTargetBase & { x: number; y: number };

type EditingTarget = { kind: "project" | "chat"; id: string; name: string };

type ProjectSidebarProps = {
  projects: Project[];
  selectedProjectId?: string;
  chats: Chat[];
  onNewProject: () => void;
  onNewChat: (projectId: string) => void;
  onRenameProject: (projectId: string, name: string) => void;
  onDeleteProject: (project: Project) => void;
  onRenameChat: (chatId: string, name: string) => void;
  onDeleteChat: (chat: Chat) => void;
};

/** Render projects and chats with compact contextual editing controls. */
export function ProjectSidebar({
  projects,
  selectedProjectId,
  chats,
  onNewProject,
  onNewChat,
  onRenameProject,
  onDeleteProject,
  onRenameChat,
  onDeleteChat,
}: ProjectSidebarProps) {
  const [contextTarget, setContextTarget] = useState<ContextTarget | null>(null);
  const [editingTarget, setEditingTarget] = useState<EditingTarget | null>(null);
  const sortedChats = sortChatsByActivity(chats);

  useEffect(() => {
    if (!contextTarget) {
      return undefined;
    }
    const close = (): void => setContextTarget(null);
    window.addEventListener("pointerdown", close);
    return () => window.removeEventListener("pointerdown", close);
  }, [contextTarget]);

  const openContextMenu = (
    event: MouseEvent,
    target: ContextTargetBase,
  ): void => {
    event.preventDefault();
    event.stopPropagation();
    setContextTarget({ ...target, x: event.clientX, y: event.clientY });
  };

  const beginRename = (): void => {
    if (!contextTarget) {
      return;
    }
    setEditingTarget({
      kind: contextTarget.kind,
      id: contextTarget.item.id,
      name: contextTarget.item.name,
    });
    setContextTarget(null);
  };

  const confirmRename = (name: string): void => {
    if (!editingTarget || !name.trim()) {
      return;
    }
    if (editingTarget.kind === "project") {
      onRenameProject(editingTarget.id, name.trim());
    } else {
      onRenameChat(editingTarget.id, name.trim());
    }
    setEditingTarget(null);
  };

  const requestDelete = (): void => {
    if (!contextTarget) {
      return;
    }
    if (contextTarget.kind === "project") {
      onDeleteProject(contextTarget.item);
    } else {
      onDeleteChat(contextTarget.item);
    }
    setContextTarget(null);
  };

  return (
    <aside className="project-sidebar">
      <div className="sidebar-title">Projects</div>
      <nav aria-label="Projects">
        {projects.map((project) => {
          const isSelected = project.id === selectedProjectId;
          const isEditingProject =
            editingTarget?.kind === "project" && editingTarget.id === project.id;
          return (
            <section className="project-tree" key={project.id}>
              <div
                className={`project-row ${isSelected ? "selected" : ""}`}
                onContextMenu={(event) =>
                  openContextMenu(event, { kind: "project", item: project })
                }
              >
                {isEditingProject ? (
                  <InlineNameEditor
                    initialName={project.name}
                    onCancel={() => setEditingTarget(null)}
                    onConfirm={confirmRename}
                  />
                ) : (
                  <NavLink to={`/projects/${project.id}/sources`}>
                    {project.name}
                  </NavLink>
                )}
                <IconButton
                  aria-label={`Create a chat in ${project.name}`}
                  onClick={() => onNewChat(project.id)}
                  title="New chat"
                >
                  ＋
                </IconButton>
              </div>
              {isSelected && sortedChats.length > 0 && (
                <div className="chat-tree">
                  {sortedChats.map((chat) => {
                    const isEditingChat =
                      editingTarget?.kind === "chat" && editingTarget.id === chat.id;
                    return (
                      <div
                        className="chat-row-container"
                        key={chat.id}
                        onContextMenu={(event) =>
                          openContextMenu(event, { kind: "chat", item: chat })
                        }
                      >
                        {isEditingChat ? (
                          <InlineNameEditor
                            initialName={chat.name}
                            onCancel={() => setEditingTarget(null)}
                            onConfirm={confirmRename}
                          />
                        ) : (
                          <NavLink
                            to={`/projects/${project.id}/chats/${chat.id}`}
                            className={({ isActive }) =>
                              `chat-row ${isActive ? "selected" : ""}`
                            }
                          >
                            {chat.name}
                          </NavLink>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
            </section>
          );
        })}
      </nav>
      <Button variant="primary" className="new-project" onClick={onNewProject}>
        ＋ New project
      </Button>
      {contextTarget && (
        <div
          aria-label={`${contextTarget.kind} actions`}
          className="sidebar-context-menu"
          onPointerDown={(event) => event.stopPropagation()}
          role="menu"
          style={{ left: contextTarget.x, top: contextTarget.y }}
        >
          <button onClick={beginRename} role="menuitem" type="button">
            Rename
          </button>
          <button
            className="danger-action"
            onClick={requestDelete}
            role="menuitem"
            type="button"
          >
            Delete
          </button>
        </div>
      )}
    </aside>
  );
}

function InlineNameEditor({
  initialName,
  onCancel,
  onConfirm,
}: {
  initialName: string;
  onCancel: () => void;
  onConfirm: (name: string) => void;
}) {
  const [name, setName] = useState(initialName);

  const submit = (event: FormEvent): void => {
    event.preventDefault();
    onConfirm(name);
  };

  return (
    <form className="sidebar-name-editor" onSubmit={submit}>
      <input
        aria-label="Rename"
        autoFocus
        maxLength={255}
        onChange={(event) => setName(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Escape") {
            event.preventDefault();
            onCancel();
          }
        }}
        value={name}
      />
      <button aria-label="Confirm rename" disabled={!name.trim()} type="submit">
        ✓
      </button>
    </form>
  );
}
