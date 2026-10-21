import { request } from "./client";
import type { ModelCatalog, Pipeline, PipelineCatalog, PipelineConfig } from "./types";

export const pipelinesApi = {
  list: (includeInactive = false) =>
    request<Pipeline[]>(
      `/pipelines${includeInactive ? "?include_inactive=true" : ""}`,
    ),
  get: (pipelineId: string) => request<Pipeline>(`/pipelines/${pipelineId}`),
  create: (config: PipelineConfig) =>
    request<Pipeline>("/pipelines", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    }),
  update: (pipelineId: string, config: PipelineConfig) =>
    request<Pipeline>(`/pipelines/${pipelineId}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    }),
  deactivate: (pipelineId: string) =>
    request<void>(`/pipelines/${pipelineId}`, { method: "DELETE" }),
  catalog: () => request<PipelineCatalog>("/catalog"),
  models: () => request<ModelCatalog>("/models"),
};
