import { describe, expect, it } from "vitest";

import type { PipelineConfig } from "../../api/types";
import { orderPipelineStages, QUERY_STAGES } from "./PipelineEditor";

describe("pipeline query stage order", () => {
  it("keeps assessors and decision policy immediately before the reasoner", () => {
    expect(QUERY_STAGES.map((stage) => stage.key)).toEqual([
      "retriever",
      "reranker",
      "assessors",
      "decision_policy",
      "reasoner",
      "renderer",
    ]);
  });

  it("serializes query stages in canonical order without dropping other settings", () => {
    const config: PipelineConfig = {
      schema_version: 1,
      kind: "query",
      name: "query",
      version: 1,
      description: "",
      is_default: false,
      query: {
        reasoner: { type: "reasoner" },
        renderer: { type: "renderer" },
        retriever: { type: "retriever" },
        custom_setting: true,
        assessors: [{ type: "assessor" }],
        decision_policy: { type: "decision_policy" },
        reranker: { type: "reranker" },
      },
    };

    const ordered = orderPipelineStages(config);

    expect(Object.keys(ordered.query ?? {})).toEqual([
      "retriever",
      "reranker",
      "assessors",
      "decision_policy",
      "reasoner",
      "renderer",
      "custom_setting",
    ]);
    expect(ordered.query?.custom_setting).toBe(true);
  });
});
