import type { PropsWithChildren, ReactNode } from "react";

type AppShellProps = PropsWithChildren<{
  sidebarOpen: boolean;
  sidebar: ReactNode;
  header: ReactNode;
  evidence: ReactNode;
  evidenceOpen: boolean;
  onSidebarToggle: () => void;
  onEvidenceToggle: () => void;
}>;

/** Arrange the optional sidebar, main workspace, and evidence panel. */
export function AppShell({
  sidebarOpen,
  sidebar,
  header,
  evidence,
  evidenceOpen,
  onSidebarToggle,
  onEvidenceToggle,
  children,
}: AppShellProps) {
  const sidebarToggle = (
    <button
      aria-label={sidebarOpen ? "Hide navigation" : "Show navigation"}
      className={`panel-toggle sidebar-toggle ${sidebarOpen ? "open" : "closed"}`}
      onClick={onSidebarToggle}
      title={sidebarOpen ? "Hide navigation" : "Show navigation"}
      type="button"
    >
      {sidebarOpen ? "‹" : "›"}
    </button>
  );
  const evidenceToggle = (
    <button
      aria-label={evidenceOpen ? "Hide evidence" : "Show evidence"}
      className={`panel-toggle evidence-toggle ${evidenceOpen ? "open" : "closed"}`}
      onClick={onEvidenceToggle}
      title={evidenceOpen ? "Hide evidence" : "Show evidence"}
      type="button"
    >
      {evidenceOpen ? "›" : "‹"}
    </button>
  );

  return (
    <div
      className={`app-shell ${sidebarOpen ? "with-sidebar" : ""} ${evidenceOpen ? "with-evidence" : ""}`}
    >
      {sidebarOpen ? (
        <div className="sidebar-slot">
          {sidebar}
          {sidebarToggle}
        </div>
      ) : (
        sidebarToggle
      )}
      <main className="main-workspace">
        <div className="header-slot">{header}</div>
        {children}
      </main>
      {evidenceOpen ? (
        <aside className="evidence-slot">
          {evidence}
          {evidenceToggle}
        </aside>
      ) : (
        evidenceToggle
      )}
    </div>
  );
}
