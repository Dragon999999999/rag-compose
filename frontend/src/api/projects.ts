import { request } from "./client";
import type { Project } from "./types";
const json = (name: string) => ({ method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name }) });
export const projectsApi = { list: () => request<Project[]>("/projects"), create: (name: string) => request<Project>("/projects", json(name)), rename: (id: string, name: string) => request<Project>(`/projects/${id}`, { ...json(name), method: "PATCH" }), remove: (id: string) => request<void>(`/projects/${id}`, { method: "DELETE" }) };
