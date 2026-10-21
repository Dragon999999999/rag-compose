import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { chatsApi } from "../api/chats";
import { documentsApi } from "../api/documents";
import { pipelinesApi } from "../api/pipelines";
import { projectsApi } from "../api/projects";
import { ragApi } from "../api/rag";
import type { Message, PipelineConfig, PipelineKind } from "../api/types";
import { findReusableDraftChat } from "./chatNavigation";

export const keys = { projects: ["projects"] as const, pipelines: (includeInactive = false, kind?: PipelineKind) => ["pipelines", { includeInactive, kind }] as const, chats: (projectId: string) => ["projects", projectId, "chats"] as const, messages: (chatId: string) => ["chats", chatId, "messages"] as const, sources: (projectId: string) => ["projects", projectId, "sources"] as const, documentSummary: (projectId: string, documentId: string) => ["projects", projectId, "documents", documentId, "summary"] as const, evidence: (runId: string) => ["runs", runId, "evidence"] as const };
export const useProjects = () => useQuery({ queryKey: keys.projects, queryFn: projectsApi.list });
export const usePipelines = (includeInactive = false, kind?: PipelineKind) => useQuery({ queryKey: keys.pipelines(includeInactive, kind), queryFn: () => pipelinesApi.list(includeInactive, kind) });
export const useCompatibleQueryPipelines = (projectId?: string) => useQuery({ queryKey: ["projects", projectId, "compatible-query-pipelines"], queryFn: () => projectsApi.compatibleQueryPipelines(projectId!), enabled: Boolean(projectId) });
export const usePipeline = (pipelineId?: string) => useQuery({ queryKey: ["pipelines", pipelineId], queryFn: () => pipelinesApi.get(pipelineId!), enabled: Boolean(pipelineId) });
export const usePipelineCatalog = () => useQuery({ queryKey: ["pipeline-catalog"], queryFn: pipelinesApi.catalog });
export const useModelCatalog = () => useQuery({ queryKey: ["model-catalog"], queryFn: pipelinesApi.models });
export const useChats = (projectId?: string) => useQuery({ queryKey: keys.chats(projectId ?? ""), queryFn: () => chatsApi.list(projectId!), enabled: !!projectId });
export const useMessages = (chatId?: string) => useQuery({ queryKey: keys.messages(chatId ?? ""), queryFn: () => chatsApi.messages(chatId!), enabled: !!chatId });
export const useSources = (projectId?: string) => useQuery({ queryKey: keys.sources(projectId ?? ""), queryFn: () => documentsApi.list(projectId!), enabled: !!projectId });
export const useDocumentSummary = (projectId?: string, documentId?: string) => useQuery({ queryKey: keys.documentSummary(projectId ?? "", documentId ?? ""), queryFn: () => documentsApi.summary(projectId!, documentId!), enabled: !!projectId && !!documentId });
export const useEvidence = (runId?: string) => useQuery({ queryKey: keys.evidence(runId ?? ""), queryFn: () => ragApi.evidence(runId!), enabled: !!runId });
export function useProjectMutations() { const client = useQueryClient(); return { create: useMutation({ mutationFn: ({ name, ingestionPipelineId }: { name: string; ingestionPipelineId?: string }) => projectsApi.create(name, ingestionPipelineId), onSuccess: () => client.invalidateQueries({ queryKey: keys.projects }) }), rename: useMutation({ mutationFn: ({ id, name }: { id: string; name: string }) => projectsApi.rename(id, name), onSuccess: () => client.invalidateQueries({ queryKey: keys.projects }) }), updateIngestion: useMutation({ mutationFn: ({ id, ingestionPipelineId }: { id: string; ingestionPipelineId: string }) => projectsApi.updateIngestion(id, ingestionPipelineId), onSuccess: (_project, variables) => { void client.invalidateQueries({ queryKey: keys.projects }); void client.invalidateQueries({ queryKey: ["projects", variables.id, "compatible-query-pipelines"] }); } }), remove: useMutation({ mutationFn: projectsApi.remove, onSuccess: () => client.invalidateQueries({ queryKey: keys.projects }) }) }; }
export function useChatMutations(projectId?: string, chatId?: string) { const client = useQueryClient(); return { create: useMutation({ mutationFn: async ({ projectId: targetProjectId, name, pipelineId }: { projectId: string; name: string; pipelineId?: string }) => { const chats = await chatsApi.list(targetProjectId); const reusable = findReusableDraftChat(chats, targetProjectId); if (reusable) { return pipelineId && reusable.pipeline_id !== pipelineId ? chatsApi.updatePipeline(reusable.id, pipelineId) : reusable; } return chatsApi.create(targetProjectId, name, pipelineId); }, onSuccess: (_, variables) => client.invalidateQueries({ queryKey: keys.chats(variables.projectId) }) }), rename: useMutation({ mutationFn: ({ chatId: targetChatId, name }: { chatId: string; name: string }) => chatsApi.rename(targetChatId, name), onSuccess: () => projectId && client.invalidateQueries({ queryKey: keys.chats(projectId) }) }), updatePipeline: useMutation({ mutationFn: ({ chatId: targetChatId, pipelineId }: { chatId: string; pipelineId: string }) => chatsApi.updatePipeline(targetChatId, pipelineId), onSuccess: () => projectId && client.invalidateQueries({ queryKey: keys.chats(projectId) }) }), remove: useMutation({ mutationFn: (targetChatId: string) => chatsApi.remove(targetChatId), onSuccess: () => projectId && client.invalidateQueries({ queryKey: keys.chats(projectId) }) }), ask: useMutation({ mutationFn: ({ message }: { message: string }) => chatsApi.ask(chatId!, message), onMutate: async ({ message }) => { if (!chatId) return { previousMessages: undefined }; const messageKey = keys.messages(chatId); await client.cancelQueries({ queryKey: messageKey }); const previousMessages = client.getQueryData<Message[]>(messageKey); const optimisticMessage: Message = { id: `pending-${Date.now()}`, chat_id: chatId, role: "user", content: message, created_at: new Date().toISOString(), rag_run_id: null, turn_title: null }; client.setQueryData<Message[]>(messageKey, (messages) => [...(messages ?? []), optimisticMessage]); return { previousMessages }; }, onError: (_error, _variables, context) => { if (chatId) client.setQueryData(keys.messages(chatId), context?.previousMessages); }, onSuccess: () => { if (projectId) client.invalidateQueries({ queryKey: keys.chats(projectId) }); }, onSettled: () => { if (chatId) client.invalidateQueries({ queryKey: keys.messages(chatId) }); } }) }; }
export function useSourceMutations(projectId?: string) { const client = useQueryClient(); return { upload: useMutation({ mutationFn: (file: File) => documentsApi.upload(projectId!, file), onSuccess: () => client.invalidateQueries({ queryKey: keys.sources(projectId!) }) }), remove: useMutation({ mutationFn: (documentId: string) => documentsApi.remove(projectId!, documentId), onSuccess: () => client.invalidateQueries({ queryKey: keys.sources(projectId!) }) }) }; }
export function useDocumentSummaryMutation(
  projectId?: string,
  documentId?: string,
) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: () => documentsApi.regenerateSummary(projectId!, documentId!),
    onSuccess: (summary) => {
      if (!projectId || !documentId) {
        return;
      }
      client.setQueryData(keys.documentSummary(projectId, documentId), summary);
      client.invalidateQueries({ queryKey: keys.sources(projectId) });
    },
  });
}

export function usePipelineMutations() {
  const client = useQueryClient();
  const invalidate = (): void => {
    void client.invalidateQueries({ queryKey: ["pipelines"] });
  };
  return {
    create: useMutation({
      mutationFn: pipelinesApi.create,
      onSuccess: invalidate,
    }),
    update: useMutation({
      mutationFn: ({ pipelineId, config, expectedConfigHash, migrateConversations }: { pipelineId: string; config: PipelineConfig; expectedConfigHash?: string; migrateConversations?: boolean }) =>
        pipelinesApi.update(pipelineId, config, { expectedConfigHash, migrateConversations }),
      onSuccess: invalidate,
    }),
    impact: useMutation({
      mutationFn: ({ pipelineId, config }: { pipelineId: string; config: PipelineConfig }) =>
        pipelinesApi.impact(pipelineId, config),
    }),
    deactivate: useMutation({
      mutationFn: pipelinesApi.deactivate,
      onSuccess: invalidate,
    }),
    setDefault: useMutation({
      mutationFn: async (pipelineId: string) => {
        const pipeline = await pipelinesApi.get(pipelineId);
        if (!pipeline.config) {
          throw new Error("The complete pipeline configuration is unavailable.");
        }
        return pipelinesApi.update(pipelineId, {
          ...pipeline.config,
          is_default: true,
        });
      },
      onSuccess: invalidate,
    }),
  };
}
