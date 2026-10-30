import { describe, expect, it } from "vitest";

import type { Source } from "../../api/types";
import { mergePendingSource } from "./SourcesWorkspace";

const pending = {
  id: "uploading-1",
  name: "report.pdf",
  createdAt: "2026-10-28T10:00:00.000Z",
};

function source(id: string, name: string, status: string, createdAt: string): Source {
  return {
    document: { id, name, created_at: createdAt },
    ingestion: {
      id: `ingestion-${id}`,
      status,
      error: null,
      created_at: createdAt,
    },
  };
}

describe("mergePendingSource", () => {
  it("replaces the optimistic row with the refreshed embedding row", () => {
    const result = mergePendingSource(
      [
        source(
          "document-1",
          "report.pdf",
          "embedding",
          "2026-10-28T10:00:01.000Z",
        ),
      ],
      pending,
    );

    expect(result).toHaveLength(1);
    expect(result[0]?.document.id).toBe("document-1");
    expect(result[0]?.ingestion?.status).toBe("embedding");
  });

  it("keeps unrelated sources while an upload is pending", () => {
    const result = mergePendingSource(
      [source("document-2", "other.pdf", "ready", "2026-10-27T10:00:00.000Z")],
      pending,
    );

    expect(result.map((item) => item.document.id)).toEqual([
      "uploading-1",
      "document-2",
    ]);
  });

  it("uses the upload response while the source query is refreshing", () => {
    const uploaded = source(
      "document-1",
      "report.pdf",
      "embedding",
      "2026-10-28T10:00:01.000Z",
    );

    const result = mergePendingSource([], { ...pending, source: uploaded });

    expect(result).toEqual([uploaded]);
  });
});
