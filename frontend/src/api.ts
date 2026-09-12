export type Project = { id: string; name: string; created_at: string };
export type Chat = { id: string; project_id: string; name: string; created_at: string };
export type Location = { page: number; bounding_boxes: { x: number; y: number; width: number; height: number }[]; positioning: "exact" | "page_only" };
export type Evidence = { chunk_id: string; document_id: string; document_name: string; page_start: number | null; page_end: number | null; excerpt: string; locations: Location[] };
export type Claim = { id: string; text: string; citations: { chunk_id: string }[] };
export type GroundedAnswer = { message_id?: string; rag_run_id: string; answer_text: string; insufficient_evidence: boolean; claims: Claim[]; evidence: Record<string, Evidence>; retrieved_chunk_ids: string[] };
export type Message = { id: string; chat_id: string; role: string; content: string; created_at: string; rag_run_id: string | null };
export type Source = { document: { id: string; name: string; created_at: string }; ingestion: { status: string; error: string | null } | null };

const base = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000/api";
async function request<T>(path: string, init?: RequestInit): Promise<T> { const response = await fetch(`${base}${path}`, init); if (!response.ok) throw new Error((await response.text()) || `Request failed (${response.status})`); return response.status === 204 ? undefined as T : response.json() as Promise<T>; }
export const api = {
  projects: () => request<Project[]>("/projects"), createProject: (name: string) => request<Project>("/projects", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name }) }),
  chats: (projectId: string) => request<Chat[]>(`/projects/${projectId}/chats`), createChat: (projectId: string, name: string) => request<Chat>(`/projects/${projectId}/chats`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name }) }),
  messages: (chatId: string) => request<Message[]>(`/chats/${chatId}/messages`), ask: (chatId: string, message: string, summary = false) => request<GroundedAnswer>(`/chats/${chatId}/messages`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ message, summary }) }), runEvidence: (runId: string) => request<GroundedAnswer>(`/runs/${runId}/evidence`),
  sources: (projectId: string) => request<Source[]>(`/projects/${projectId}/documents`), upload: (projectId: string, file: File) => { const body = new FormData(); body.set("file", file); return request<Source>(`/projects/${projectId}/documents`, { method: "POST", body }); }, pdf: (documentId: string) => `${base}/documents/${documentId}/pdf`
};
