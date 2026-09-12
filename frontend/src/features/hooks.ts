import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { chatsApi } from "../api/chats";
import { documentsApi } from "../api/documents";
import { projectsApi } from "../api/projects";
import { ragApi } from "../api/rag";

export const keys = { projects: ["projects"] as const, chats: (projectId: string) => ["projects", projectId, "chats"] as const, messages: (chatId: string) => ["chats", chatId, "messages"] as const, sources: (projectId: string) => ["projects", projectId, "sources"] as const, evidence: (runId: string) => ["runs", runId, "evidence"] as const };
export const useProjects = () => useQuery({ queryKey: keys.projects, queryFn: projectsApi.list });
export const useChats = (projectId?: string) => useQuery({ queryKey: keys.chats(projectId ?? ""), queryFn: () => chatsApi.list(projectId!), enabled: !!projectId });
export const useMessages = (chatId?: string) => useQuery({ queryKey: keys.messages(chatId ?? ""), queryFn: () => chatsApi.messages(chatId!), enabled: !!chatId });
export const useSources = (projectId?: string) => useQuery({ queryKey: keys.sources(projectId ?? ""), queryFn: () => documentsApi.list(projectId!), enabled: !!projectId });
export const useEvidence = (runId?: string) => useQuery({ queryKey: keys.evidence(runId ?? ""), queryFn: () => ragApi.evidence(runId!), enabled: !!runId });
export function useProjectMutations() { const client = useQueryClient(); return { create: useMutation({ mutationFn: projectsApi.create, onSuccess: () => client.invalidateQueries({ queryKey: keys.projects }) }), rename: useMutation({ mutationFn: ({ id, name }: { id: string; name: string }) => projectsApi.rename(id, name), onSuccess: () => client.invalidateQueries({ queryKey: keys.projects }) }), remove: useMutation({ mutationFn: projectsApi.remove, onSuccess: () => client.invalidateQueries({ queryKey: keys.projects }) }) }; }
export function useChatMutations(projectId?: string, chatId?: string) { const client = useQueryClient(); return { create: useMutation({ mutationFn: (name: string) => chatsApi.create(projectId!, name), onSuccess: () => client.invalidateQueries({ queryKey: keys.chats(projectId!) }) }), ask: useMutation({ mutationFn: ({ message, summary }: { message: string; summary: boolean }) => chatsApi.ask(chatId!, message, summary), onSuccess: () => client.invalidateQueries({ queryKey: keys.messages(chatId!) }) }) }; }
export function useSourceMutations(projectId?: string) { const client = useQueryClient(); return { upload: useMutation({ mutationFn: (file: File) => documentsApi.upload(projectId!, file), onSuccess: () => client.invalidateQueries({ queryKey: keys.sources(projectId!) }) }), remove: useMutation({ mutationFn: (documentId: string) => documentsApi.remove(projectId!, documentId), onSuccess: () => client.invalidateQueries({ queryKey: keys.sources(projectId!) }) }) }; }
