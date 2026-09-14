import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import type { DocumentSummary, Source } from "../../api/types";
import { DocumentOverview } from "./DocumentOverview";

const source: Source = {
  document: { id: "document", name: "Kumpf2024.pdf", created_at: "2026-09-14" },
  ingestion: { id: "ingestion", status: "ready", error: null, created_at: "2026-09-14" },
  summary: null,
};

const summary: DocumentSummary = {
  document_id: "document",
  ingestion_id: "ingestion",
  status: "ready",
  rag_run_id: "run",
  error: null,
  page_count: 24,
  answer: {
    rag_run_id: "run",
    turn_title: "Document overview",
    answer_text: "The paper investigates a supported topic.",
    insufficient_evidence: false,
    claims: [{ id: "claim", text: "The paper investigates a topic.", citations: [{ chunk_id: "chunk" }] }],
    evidence: {
      chunk: {
        chunk_id: "chunk",
        reference_label: "Kumpf2024_5",
        used: true,
        document_id: "document",
        document_name: "Kumpf2024.pdf",
        page_start: 3,
        page_end: 3,
        excerpt: "Supported source text.",
        locations: [{ page: 3, bounding_boxes: [], positioning: "page_only" }],
      },
    },
    retrieved_chunk_ids: ["chunk"],
  },
};

describe("DocumentOverview", () => {
  it("renders persisted grounded claims with existing-style citation labels", () => {
    const markup = renderToStaticMarkup(
      <DocumentOverview
        activeChunk={null}
        loading={false}
        onBack={() => undefined}
        onSelectCitation={() => undefined}
        source={source}
        summary={summary}
      />,
    );

    expect(markup).toContain("Kumpf2024.pdf");
    expect(markup).toContain("24 pages");
    expect(markup).toContain("The paper investigates a topic.");
    expect(markup).toContain("Kumpf2024");
    expect(markup).toContain("C5");
  });

  it("renders the persisted summary lifecycle without fabricating progress", () => {
    const markup = renderToStaticMarkup(
      <DocumentOverview
        activeChunk={null}
        loading={false}
        onBack={() => undefined}
        onSelectCitation={() => undefined}
        source={source}
        summary={{ ...summary, status: "generating", answer: undefined }}
      />,
    );

    expect(markup).toContain("Generating summary");
    expect(markup).not.toContain("%");
  });
});
