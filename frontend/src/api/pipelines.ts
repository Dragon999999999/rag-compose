import { request } from "./client";
import type { ModelCatalog, Pipeline, PipelineCatalog, PipelineConfig, PipelineKind, PipelineUpdateImpact } from "./types";

export const pipelinesApi = {
  list: (includeInactive = false, kind?: PipelineKind) => {
    const query = new URLSearchParams();
    if (includeInactive) query.set("include_inactive", "true");
    if (kind) query.set("kind", kind);
    const suffix = query.toString() ? `?${query}` : "";
    return request<Pipeline[]>(`/pipelines${suffix}`);
  },
  get: (pipelineId: string) => request<Pipeline>(`/pipelines/${pipelineId}`),
  create: (config: PipelineConfig) =>
    request<Pipeline>("/pipelines", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    }),
  impact: (pipelineId: string, config: PipelineConfig) =>
    request<PipelineUpdateImpact>(`/pipelines/${pipelineId}/impact`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    }),
  update: (pipelineId: string, config: PipelineConfig, options?: { expectedConfigHash?: string; migrateConversations?: boolean }) => {
    const query = new URLSearchParams();
    if (options?.expectedConfigHash) query.set("expected_config_hash", options.expectedConfigHash);
    if (options?.migrateConversations) query.set("migrate_conversations", "true");
    const suffix = query.toString() ? `?${query.toString()}` : "";
    return request<Pipeline>(`/pipelines/${pipelineId}${suffix}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    });
  },
  deactivate: (pipelineId: string) =>
    request<void>(`/pipelines/${pipelineId}`, { method: "DELETE" }),
  catalog: () => request<PipelineCatalog>("/catalog"),
  models: () => request<ModelCatalog>("/models"),
};
