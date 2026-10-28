import { describe, expect, it } from "vitest";

import type { PipelineAlternative, PipelineConfig } from "../../api/types";
import {
  addQueryBranch,
  addQueryAlternative,
  buildBranchGraphLayout,
  orderPipelineStages,
  QUERY_STAGES,
  removeQueryBranch,
  updateAlternativeSource,
  readExecutionStrategy,
  updateExecutionStrategy,
} from "./PipelineEditor";

function queryConfig(execution?: Record<string, unknown>): PipelineConfig {
  return {
    schema_version: 1,
    kind: "query",
    name: "query",
    version: 1,
    description: "",
    is_default: false,
    query: { execution },
  };
}

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

  it("persists execution strategy settings while retaining a valid branch shape", () => {
    const config = queryConfig();

    const branching = updateExecutionStrategy(config, "branching");

    expect(readExecutionStrategy(branching)).toBe("branching");
    expect(branching.query?.execution).toEqual({
      strategy: "branching",
      branches: [{ id: "branch-1" }, { id: "branch-2" }],
    });
  });

  it("creates the first split at the selected stage and keeps stable branch IDs", () => {
    const branching = addQueryBranch(queryConfig(), "assessor");
    const execution = branching.query?.execution as Record<string, unknown>;

    expect(execution.strategy).toBe("branching");
    expect(execution.branches).toEqual([
      { id: "branch-1", start_at: "assessor" },
      { id: "branch-2", start_at: "assessor" },
    ]);
  });

  it("centers shared and merged stages from the descendant branch rows", () => {
    const config = queryConfig({
      strategy: "branching",
      branches: [
        { id: "branch-a", start_at: "assessor" },
        { id: "branch-b", start_at: "assessor" },
        { id: "branch-c", start_at: "assessor" },
      ],
    });
    const layout = buildBranchGraphLayout(config);
    const decision = layout.nodes.find((node) => node.key === "shared-decision_policy");
    const retriever = layout.nodes.find((node) => node.key === "shared-retriever");
    const branchCenters = layout.branches.map((branch) => branch.center);

    expect(decision?.y).toBeCloseTo(branchCenters[1] - 36);
    expect(retriever?.y).toBeCloseTo(branchCenters[1] - 36);
    expect(layout.nodes.filter((node) => node.definition.key === "assessors")).toHaveLength(3);
  });

  it("supports later splits and returns to linear execution after the last branch is removed", () => {
    const firstSplit = addQueryBranch(queryConfig(), "retriever");
    const nestedSplit = addQueryBranch(firstSplit, "reranker", "branch-1");
    const removed = removeQueryBranch(nestedSplit, "branch-2");
    const execution = removed.query?.execution as Record<string, unknown>;

    expect((nestedSplit.query?.execution as Record<string, unknown>).branches).toHaveLength(3);
    expect(execution.strategy).toBe("branching");
    expect(execution.branches).toHaveLength(2);
    expect(removeQueryBranch(removed, "branch-1").query?.execution).toMatchObject({ strategy: "linear" });
  });

  it("creates alternatives from the latest effective path and updates restart selection", () => {
    const primary = addQueryBranch(queryConfig(), "retriever", undefined, "alternative-1");
    const withAlternative = addQueryAlternative(primary);
    const execution = withAlternative.query?.execution as Record<string, unknown>;
    const alternatives = execution.alternatives as Array<Record<string, unknown>>;
    const retry = alternatives[1];

    expect(retry.source).toEqual({
      alternative_id: "alternative-1",
      path_id: "branch-1",
      restart_stage: "assessor",
    });
    const rerankerRetry = updateAlternativeSource(withAlternative, "alternative-2", "reranker", "branch-1");
    const updated = ((rerankerRetry.query?.execution as Record<string, unknown>).alternatives as Array<Record<string, unknown>>)[1];
    expect(updated.source).toMatchObject({ restart_stage: "reranker", path_id: "branch-1" });
    expect((updated.branches as Array<Record<string, unknown>>)[0].start_at).toBe("reranker");
  });

  it("marks upstream nodes as reused in an alternative graph", () => {
    const config = addQueryAlternative(queryConfig());
    const execution = config.query?.execution as Record<string, unknown>;
    const alternatives = execution.alternatives as Array<Record<string, unknown>>;
    const layout = buildBranchGraphLayout(config, alternatives[1] as unknown as PipelineAlternative);

    expect(layout.nodes.find((node) => node.definition.key === "retriever")?.reused).toBe(true);
    expect(layout.nodes.find((node) => node.definition.key === "assessors")?.reused).toBe(false);
  });

  it("keeps alternative topology separate from the shared decision policy", () => {
    const config = addQueryAlternative(queryConfig());
    const alternatives = (config.query?.execution as Record<string, unknown>).alternatives as Array<Record<string, unknown>>;

    expect(alternatives[1]).not.toHaveProperty("acceptance");
    expect(alternatives[1]).not.toHaveProperty("decision_policy");
  });
});
