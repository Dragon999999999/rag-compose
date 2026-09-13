import { useEffect, useRef } from "react";

import type { Evidence, GroundedAnswer } from "../../api/types";
import { Button, EmptyState } from "../common/ui";
import { PdfViewer } from "../pdf/PdfViewer";

type EvidenceWorkspaceProps = {
  answer: GroundedAnswer | null;
  activeChunk: string | null;
  mode: "evidence" | "pdf";
  documentEvidence: Evidence | null;
  turnTitle?: string | null;
  onSelect: (id: string) => void;
  onOpenPdf: (evidence: Evidence) => void;
  onBack: () => void;
};

/** Show every retrieved chunk for the prompt-answer turn currently in focus. */
export function EvidenceWorkspace({
  answer,
  activeChunk,
  mode,
  documentEvidence,
  turnTitle,
  onSelect,
  onOpenPdf,
  onBack,
}: EvidenceWorkspaceProps) {
  const workspace = useRef<HTMLElement>(null);
  const evidenceCards = useRef(new Map<string, HTMLElement>());
  const active = activeChunk && answer?.evidence[activeChunk];
  const pdfEvidence = documentEvidence ?? active;

  useEffect(() => {
    if (mode !== "evidence" || !activeChunk) {
      return;
    }
    const container = workspace.current;
    const card = evidenceCards.current.get(activeChunk);
    if (!container || !card) {
      return;
    }
    const top =
      card.getBoundingClientRect().top -
      container.getBoundingClientRect().top +
      container.scrollTop -
      12;
    container.scrollTo({ top: Math.max(top, 0), behavior: "smooth" });
  }, [activeChunk, answer?.rag_run_id, mode]);

  if (mode === "pdf" && pdfEvidence) {
    return <PdfViewer evidence={pdfEvidence} onBack={onBack} />;
  }
  if (!answer) {
    return (
      <section className="evidence-workspace" ref={workspace}>
        <EmptyState title="Evidence workspace">
          Select an assistant response or conversation turn to inspect its retrieved
          evidence.
        </EmptyState>
      </section>
    );
  }

  const retrieved = Object.values(answer.evidence);
  const citedChunkIds = new Set(
    answer.claims.flatMap((claim) =>
      claim.citations.map((citation) => citation.chunk_id),
    ),
  );
  const usedCount = retrieved.filter((evidence) =>
    citedChunkIds.has(evidence.chunk_id),
  ).length;
  return (
    <section className="evidence-workspace" ref={workspace}>
      <header>
        <div className="evidence-heading">
          <h2>Retrieved evidence</h2>
          {turnTitle && <span className="turn-evidence-tag">{turnTitle}</span>}
        </div>
        <p>
          <strong>{usedCount} used</strong> · {retrieved.length - usedCount} additional
          retrieved · {retrieved.length} total
        </p>
      </header>
      <div className="evidence-list">
        {retrieved.map((evidence) => {
          const used = citedChunkIds.has(evidence.chunk_id);
          return (
            <article
              className={`evidence-card ${
                used ? "used" : "retrieved-only"
              } ${activeChunk === evidence.chunk_id ? "selected" : ""}`}
              key={evidence.chunk_id}
              onClick={() => onSelect(evidence.chunk_id)}
              ref={(element) => {
                if (element) {
                  evidenceCards.current.set(evidence.chunk_id, element);
                } else {
                  evidenceCards.current.delete(evidence.chunk_id);
                }
              }}
            >
              <div className="evidence-card-title">
                <strong>{evidence.reference_label}</strong>
                <span>p.{evidence.page_start ?? "?"}</span>
              </div>
              <small className="evidence-document-name">
                {evidence.document_name}
              </small>
              <div className="evidence-crop">
                <mark>{evidence.excerpt}</mark>
              </div>
              <small>
                {used
                  ? "Cited in this response"
                  : "Retrieved for this response, not cited"}
                {evidence.locations.some(
                  (location) => location.positioning === "exact",
                )
                  ? " · Exact PDF location"
                  : " · Page-level location"}
              </small>
              <Button
                onClick={(event) => {
                  event.stopPropagation();
                  onOpenPdf(evidence);
                }}
              >
                Open PDF
              </Button>
            </article>
          );
        })}
      </div>
    </section>
  );
}
