import { request } from "./client";
import type { PipelineCompatibility, Project } from "./types";
const headers = { "Content-Type": "application/json" };
export const projectsApi = {
  list: () => request<Project[]>("/projects"),
  create: (name: string, ingestionPipelineId?: string) =>
    request<Project>("/projects", {
      method: "POST",
      headers,
      body: JSON.stringify({ name, ...(ingestionPipelineId ? { ingestion_pipeline_id: ingestionPipelineId } : {}) }),
    }),
  rename: (id: string, name: string) =>
    request<Project>(`/projects/${id}`, { method: "PATCH", headers, body: JSON.stringify({ name }) }),
  updateIngestion: (id: string, ingestionPipelineId: string) =>
    request<Project>(`/projects/${id}`, { method: "PATCH", headers, body: JSON.stringify({ ingestion_pipeline_id: ingestionPipelineId }) }),
  compatibleQueryPipelines: (id: string) =>
    request<PipelineCompatibility[]>(`/projects/${id}/compatible-query-pipelines`),
  remove: (id: string) => request<void>(`/projects/${id}`, { method: "DELETE" }),
};
