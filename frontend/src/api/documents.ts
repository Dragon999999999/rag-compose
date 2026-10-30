import { request } from "./client";
import type { Source } from "./types";

export const documentsApi = {
  list: (projectId: string) => request<Source[]>(`/projects/${projectId}/documents`),
  upload: (projectId: string, file: File) => {
    const body = new FormData();
    body.set("file", file);
    return request<Source>(`/projects/${projectId}/documents`, { method: "POST", body });
  },
  remove: (projectId: string, documentId: string) =>
    request<void>(`/projects/${projectId}/documents/${documentId}`, { method: "DELETE" }),
};
