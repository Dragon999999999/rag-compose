import type { DocumentSummary, GroundedAnswer, Source } from "../../api/types";
import { Button, EmptyState, ErrorState } from "../common/ui";

type DocumentOverviewProps = {
  source?: Source;
  summary?: DocumentSummary;
  loading: boolean;
  error?: Error | null;
  activeChunk: string | null;
  onBack: () => void;
  onSelectCitation: (chunkId: string) => void;
};

/** Present a document's persisted grounded overview without duplicating evidence UI. */
export function DocumentOverview({
  source,
  summary,
  loading,
  error,
  activeChunk,
  onBack,
  onSelectCitation,
}: DocumentOverviewProps) {
  const pageCount = summary?.page_count ?? source?.summary?.page_count;

  return (
    <section className="document-overview">
      <header>
        <Button onClick={onBack} variant="ghost">
          ← Back to chat
        </Button>
        <h2>{source?.document.name ?? "Document"}</h2>
        {pageCount !== null && pageCount !== undefined && <p>{pageCount} pages</p>}
      </header>
      <div className="document-overview-content">
        <h3>Summary</h3>
        {error && <ErrorState error={error} />}
        {loading && <p className="document-summary-state">Loading summary…</p>}
        {!loading && !error && summary?.status === "pending" && (
          <p className="document-summary-state">Summary pending</p>
        )}
        {!loading && !error && summary?.status === "generating" && (
          <p className="document-summary-state">Generating summary</p>
        )}
        {!loading && !error && summary?.status === "failed" && (
          <EmptyState title="Summary failed">
            {summary.error
              ? "The grounded summary could not be generated. The source PDF remains available."
              : "The grounded summary could not be generated."}
          </EmptyState>
        )}
        {!loading && !error && summary?.status === "ready" && summary.answer && (
          <SummaryClaims
            activeChunk={activeChunk}
            answer={summary.answer}
            onSelectCitation={onSelectCitation}
          />
        )}
      </div>
    </section>
  );
}

function SummaryClaims({
  answer,
  activeChunk,
  onSelectCitation,
}: {
  answer: GroundedAnswer;
  activeChunk: string | null;
  onSelectCitation: (chunkId: string) => void;
}) {
  if (answer.insufficient_evidence) {
    return (
      <article className="document-summary-insufficient">
        <strong>Insufficient evidence in this document</strong>
        <p>{answer.answer_text}</p>
      </article>
    );
  }
  return (
    <article className="document-summary-claims structured-answer">
      {answer.claims.map((claim) => (
        <p
          className={
            claim.citations.some((citation) => citation.chunk_id === activeChunk)
              ? "active-claim"
              : ""
          }
          key={claim.id}
        >
          {claim.text}{" "}
          <span className="citation-group">
            {claim.citations.map((citation) => {
              const evidence = answer.evidence[citation.chunk_id];
              return evidence ? (
                <button
                  className={`citation ${
                    citation.chunk_id === activeChunk ? "citation-related" : ""
                  }`}
                  key={citation.chunk_id}
                  onClick={() => onSelectCitation(citation.chunk_id)}
                  title={evidence.excerpt}
                  type="button"
                >
                  <CitationBadgeLabel label={evidence.reference_label} />
                </button>
              ) : null;
            })}
          </span>
        </p>
      ))}
    </article>
  );
}

function CitationBadgeLabel({ label }: { label: string }) {
  const match = /^(.*)_(\d+)$/.exec(label);
  if (!match) {
    return <span className="citation-document">{label}</span>;
  }
  return (
    <>
      <span className="citation-document">{match[1]}</span>
      <span className="citation-chunk">C{match[2]}</span>
    </>
  );
}
