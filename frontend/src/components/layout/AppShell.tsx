import type { PropsWithChildren, ReactNode } from "react";

type AppShellProps = PropsWithChildren<{
  sidebarOpen: boolean;
  sidebar: ReactNode;
  header: ReactNode;
  evidence: ReactNode;
  evidenceOpen: boolean;
}>;

/** Arrange the optional sidebar, main workspace, and evidence panel. */
export function AppShell({
  sidebarOpen,
  sidebar,
  header,
  evidence,
  evidenceOpen,
  children,
}: AppShellProps) {
  return (
    <div
      className={`app-shell ${sidebarOpen ? "with-sidebar" : ""} ${evidenceOpen ? "with-evidence" : ""}`}
    >
      {sidebarOpen && <div className="sidebar-slot">{sidebar}</div>}
      <main className="main-workspace">
        <div className="header-slot">{header}</div>
        {children}
      </main>
      {evidenceOpen && <aside className="evidence-slot">{evidence}</aside>}
    </div>
  );
}
