import { request } from "./client";
import type { Chat, GroundedAnswer, Message } from "./types";

const json = (name: string) => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ name }),
});

export const chatsApi = {
  list: (projectId: string) => request<Chat[]>(`/projects/${projectId}/chats`),
  create: (projectId: string, name: string) =>
    request<Chat>(`/projects/${projectId}/chats`, json(name)),
  rename: (chatId: string, name: string) =>
    request<Chat>(`/chats/${chatId}`, { ...json(name), method: "PATCH" }),
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
