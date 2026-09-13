import { useState } from "react";
import { Navigate, Route, Routes, useNavigate, useParams } from "react-router-dom";

import type { Chat, Evidence, GroundedAnswer, Project } from "../api/types";
import { ChatWorkspace } from "../components/chats/ChatWorkspace";
import { Dialog, NameDialog } from "../components/common/Dialog";
import { Button, EmptyState, ErrorState } from "../components/common/ui";
import { EvidenceWorkspace } from "../components/evidence/EvidenceWorkspace";
import { AppShell } from "../components/layout/AppShell";
import { ProjectHeader } from "../components/layout/ProjectHeader";
import { ProjectSidebar } from "../components/layout/ProjectSidebar";
import { SourcesWorkspace } from "../components/sources/SourcesWorkspace";
import {
  useChatMutations,
  useChats,
  useEvidence,
  useMessages,
  useProjectMutations,
  useProjects,
  useSourceMutations,
  useSources,
} from "../features/hooks";
import "./App.css";

type DeleteTarget =
  | { kind: "project"; item: Project }
  | { kind: "chat"; item: Chat };

export function App() {
  return (
    <Routes>
      <Route path="/" element={<Navigate to="/projects" replace />} />
      <Route path="/projects" element={<ProjectHome />} />
      <Route
        path="/projects/:projectId/chats/:chatId"
        element={<ProjectWorkspace />}
      />
      <Route
        path="/projects/:projectId/sources"
        element={<ProjectWorkspace sources />}
      />
      <Route
        path="*"
        element={
          <main className="not-found">
            <EmptyState title="Page not found">
              The requested project workspace does not exist.
            </EmptyState>
          </main>
        }
      />
    </Routes>
  );
}

function ProjectHome() {
  const projects = useProjects();
  const navigate = useNavigate();
  const [create, setCreate] = useState(false);
  const mutations = useProjectMutations();

  if (projects.isLoading) {
    return <main className="home-state">Loading projects…</main>;
  }
  if (projects.error) {
    return <ErrorState error={projects.error} retry={() => projects.refetch()} />;
  }

  const firstProject = projects.data?.[0];
  if (firstProject) {
    return <Navigate to={`/projects/${firstProject.id}/sources`} replace />;
  }

  return (
    <main className="home-state">
      <EmptyState title="No projects yet">
        Create a project to organize source documents and grounded conversations.
        <p>
          <Button variant="primary" onClick={() => setCreate(true)}>
            Create project
          </Button>
        </p>
      </EmptyState>
      {create && (
        <NameDialog
          error={mutations.create.error}
          onClose={() => setCreate(false)}
          onSubmit={(name) =>
            mutations.create.mutate(name, {
              onSuccess: (project) => {
                setCreate(false);
                navigate(`/projects/${project.id}/sources`);
              },
            })
          }
          pending={mutations.create.isPending}
          submitLabel="Create project"
          title="Create project"
        />
      )}
    </main>
  );
}

function ProjectWorkspace({ sources: sourcesRoute = false }: { sources?: boolean }) {
  const { projectId, chatId } = useParams();
  const navigate = useNavigate();
  const projects = useProjects();
  const chats = useChats(projectId);
  const sources = useSources(projectId);
  const messages = useMessages(chatId);
  const project = projects.data?.find((item) => item.id === projectId);
  const chat = chats.data?.find((item) => item.id === chatId);
  const projectMutations = useProjectMutations();
  const chatMutations = useChatMutations(projectId, chatId);
  const sourceMutations = useSourceMutations(projectId);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [evidenceOpen, setEvidenceOpen] = useState(true);
  const [createProject, setCreateProject] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<DeleteTarget | null>(null);
  const [selectedRun, setSelectedRun] = useState<string>();
  const [latest, setLatest] = useState<GroundedAnswer | null>(null);
  const evidence = useEvidence(selectedRun);
  const answer = latest ?? evidence.data ?? null;
  const [activeChunk, setActiveChunk] = useState<string | null>(null);
  const [directPdf, setDirectPdf] = useState<Evidence | null>(null);
  const [mode, setMode] = useState<"evidence" | "pdf">("evidence");

  const createChat = (targetProjectId = projectId): void => {
    if (!targetProjectId) {
      return;
    }
    chatMutations.create.mutate(
      { projectId: targetProjectId, name: "New chat" },
      {
        onSuccess: (newChat) =>
          navigate(`/projects/${targetProjectId}/chats/${newChat.id}`),
      },
    );
  };

  const ask = (message: string, summary = false): void => {
    chatMutations.ask.mutate(
      { message, summary },
      {
        onSuccess: (result) => {
          setLatest(result);
          setSelectedRun(undefined);
          setActiveChunk(Object.keys(result.evidence)[0] ?? null);
          setDirectPdf(null);
          setEvidenceOpen(true);
          setMode("evidence");
        },
      },
    );
  };

  const chooseRun = (runId: string): void => {
    setLatest(null);
    setSelectedRun(runId);
    setDirectPdf(null);
    setEvidenceOpen(true);
    setMode("evidence");
  };

  const openPdf = (item: Evidence): void => {
    setDirectPdf(null);
    setActiveChunk(item.chunk_id);
    setMode("pdf");
    setEvidenceOpen(true);
  };

  const openSourcePdf = (source: {
    document: { id: string; name: string };
  }): void => {
    setDirectPdf({
      chunk_id: source.document.id,
      document_id: source.document.id,
      document_name: source.document.name,
      page_start: null,
      page_end: null,
      excerpt: "",
      locations: [],
    });
    setMode("pdf");
    setEvidenceOpen(true);
  };

  const sidebar = (
    <ProjectSidebar
      chats={chats.data ?? []}
      onDeleteChat={(targetChat) =>
        setDeleteTarget({ kind: "chat", item: targetChat })
      }
      onDeleteProject={(targetProject) =>
        setDeleteTarget({ kind: "project", item: targetProject })
      }
      onNewChat={createChat}
      onNewProject={() => setCreateProject(true)}
      onRenameChat={(targetChatId, name) =>
        chatMutations.rename.mutate({ chatId: targetChatId, name })
      }
      onRenameProject={(targetProjectId, name) =>
        projectMutations.rename.mutate({ id: targetProjectId, name })
      }
      projects={projects.data ?? []}
      selectedProjectId={projectId}
    />
  );

  const header = (
    <ProjectHeader
      chat={chat}
      onNewChat={() => createChat()}
      onSources={() => projectId && navigate(`/projects/${projectId}/sources`)}
      onSummary={() => ask("Summarize this project", true)}
      project={project}
    />
  );

  const evidencePane = (
    <EvidenceWorkspace
      activeChunk={activeChunk}
      answer={answer}
      documentEvidence={directPdf}
      mode={mode}
      onBack={() => {
        setDirectPdf(null);
        setMode("evidence");
      }}
      onOpenPdf={openPdf}
      onSelect={(id) => {
        setDirectPdf(null);
        setActiveChunk(id);
      }}
    />
  );

  if (!projectId || (!projects.isLoading && !project)) {
    return (
      <main className="home-state">
        <EmptyState title="Project not found">
          Choose a project from the navigation.
        </EmptyState>
      </main>
    );
  }

  const deletePending =
    deleteTarget?.kind === "project"
      ? projectMutations.remove.isPending
      : chatMutations.remove.isPending;

  const confirmDelete = (): void => {
    if (!deleteTarget) {
      return;
    }
    if (deleteTarget.kind === "project") {
      projectMutations.remove.mutate(deleteTarget.item.id, {
        onSuccess: () => {
          if (deleteTarget.item.id === projectId) {
            navigate("/projects");
          }
          setDeleteTarget(null);
        },
      });
      return;
    }
    chatMutations.remove.mutate(deleteTarget.item.id, {
      onSuccess: () => {
        if (deleteTarget.item.id === chatId) {
          navigate(`/projects/${projectId}/sources`);
        }
        setDeleteTarget(null);
      },
    });
  };

  return (
    <>
      <AppShell
        evidence={evidencePane}
        evidenceOpen={evidenceOpen}
        header={header}
        onEvidenceToggle={() => setEvidenceOpen(!evidenceOpen)}
        onSidebarToggle={() => setSidebarOpen(!sidebarOpen)}
        sidebar={sidebar}
        sidebarOpen={sidebarOpen}
      >
        {sourcesRoute ? (
          <SourcesWorkspace
            error={sources.error}
            loading={sources.isLoading}
            onBack={() =>
              navigate(chatId ? `/projects/${projectId}/chats/${chatId}` : "/projects")
            }
            onOpen={openSourcePdf}
            onRemove={(id) => sourceMutations.remove.mutate(id)}
            onUpload={(file) => sourceMutations.upload.mutateAsync(file)}
            sources={sources.data}
            uploadError={sourceMutations.upload.error}
            uploading={sourceMutations.upload.isPending}
          />
        ) : chatId ? (
          <ChatWorkspace
            activeChunk={activeChunk}
            answer={latest}
            error={chatMutations.ask.error ?? messages.error}
            loading={chatMutations.ask.isPending || messages.isLoading}
            messages={messages.data}
            onAsk={(message) => ask(message)}
            onSelectAnswer={chooseRun}
            onSelectCitation={(id) => {
              setDirectPdf(null);
              setActiveChunk(id);
              setEvidenceOpen(true);
              setMode("evidence");
            }}
          />
        ) : (
          <main className="home-state">
            <EmptyState title="No chat selected">
              Create a chat to start asking grounded questions.
              <p>
                <Button variant="primary" onClick={() => createChat()}>
                  New chat
                </Button>
              </p>
            </EmptyState>
          </main>
        )}
      </AppShell>
      {createProject && (
        <NameDialog
          error={projectMutations.create.error}
          onClose={() => setCreateProject(false)}
          onSubmit={(name) =>
            projectMutations.create.mutate(name, {
              onSuccess: (newProject) => {
                setCreateProject(false);
                navigate(`/projects/${newProject.id}/sources`);
              },
            })
          }
          pending={projectMutations.create.isPending}
          submitLabel="Create project"
          title="Create project"
        />
      )}
      {deleteTarget && (
        <Dialog
          onClose={() => setDeleteTarget(null)}
          title={`Delete ${deleteTarget.kind}?`}
        >
          <p>
            {deleteTarget.kind === "project"
              ? "This removes the project, its chats, and project membership. The action cannot be undone."
              : "This removes the chat and its messages. The project sources remain available."}
          </p>
          <footer>
            <Button variant="ghost" onClick={() => setDeleteTarget(null)}>
              Cancel
            </Button>
            <Button
              disabled={deletePending}
              onClick={confirmDelete}
              variant="danger"
            >
              {deletePending ? "Deleting…" : "Delete"}
            </Button>
          </footer>
        </Dialog>
      )}
    </>
  );
}
