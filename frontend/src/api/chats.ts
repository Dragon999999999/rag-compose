import { request } from "./client";
import type { Chat, GroundedAnswer, Message } from "./types";

const json = (name: string, pipelineId?: string) => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ name, ...(pipelineId ? { pipeline_id: pipelineId } : {}) }),
});

export const chatsApi = {
  list: (projectId: string) => request<Chat[]>(`/projects/${projectId}/chats`),
  create: (projectId: string, name: string, pipelineId?: string) =>
    request<Chat>(`/projects/${projectId}/chats`, json(name, pipelineId)),
  rename: (chatId: string, name: string) =>
    request<Chat>(`/chats/${chatId}`, { ...json(name), method: "PATCH" }),
  updatePipeline: (chatId: string, pipelineId: string) =>
    request<Chat>(`/chats/${chatId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pipeline_id: pipelineId }),
    }),
  remove: (chatId: string) =>
    request<void>(`/chats/${chatId}`, { method: "DELETE" }),
  messages: (chatId: string) => request<Message[]>(`/chats/${chatId}/messages`),
  ask: (chatId: string, message: string) =>
    request<GroundedAnswer>(`/chats/${chatId}/messages`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    }),
};
