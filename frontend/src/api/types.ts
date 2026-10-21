export type Project = { id: string; name: string; created_at: string };
export type JsonSchema = {
  type?: string;
  title?: string;
  description?: string;
  default?: unknown;
  const?: unknown;
  enum?: unknown[];
  anyOf?: JsonSchema[];
  properties?: Record<string, JsonSchema>;
  required?: string[];
  minimum?: number;
  maximum?: number;
  minLength?: number;
  maxLength?: number;
};
export type CatalogStage = { implementations: string[]; schema: JsonSchema };
export type PipelineCatalog = {
  schema_version: number;
  stages: Record<string, CatalogStage>;
  compatibility: Record<string, unknown>;
};
export type ModelDefinition = Record<string, unknown> & { name?: string; model?: string; dimension?: number; normalize?: boolean };
export type ModelCatalog = {
  schema_version: number;
  embedding: Record<string, ModelDefinition>;
  reasoning: Record<string, ModelDefinition>;
};
export type PipelineConfig = {
  schema_version: number;
  name: string;
  version: number;
  description: string;
  is_default: boolean;
  representation: Record<string, unknown>;
  query: Record<string, unknown>;
};
export type Pipeline = {
  id: string;
  name: string;
  version: number;
  schema_version: number;
  description: string;
  active: boolean;
  is_default: boolean;
  config_hash: string;
  representation_hash: string;
  source_kind?: string;
  created_at: string;
  updated_at: string;
  conversation_count?: number;
  config?: PipelineConfig;
};
export type PipelineChange = {
  path: string;
  before: unknown;
  after: unknown;
  representation_affecting: boolean;
};
export type PipelineUpdateImpact = {
  pipeline_id: string;
  current_version: number;
  current_config_hash: string;
  conversation_count: number;
  representation_changed: boolean;
  compatible: boolean;
  can_migrate: boolean;
  compatibility_reason: string | null;
  changes: PipelineChange[];
};
export type Chat = { id: string; project_id: string; name: string; created_at: string; chat_name_set: boolean; last_answer_at: string | null; has_sent_prompt: boolean; pipeline_id: string | null; pipeline_version: number | null; pipeline_config_hash: string | null; pipeline_representation_hash: string | null };
export type Message = { id: string; chat_id: string; role: "user" | "assistant" | "system"; content: string; created_at: string; rag_run_id: string | null; turn_title: string | null };
export type SourceLocation = { page: number; bounding_boxes: BoundingBox[]; positioning: "exact" | "page_only" };
export type BoundingBox = { x: number; y: number; width: number; height: number };
export type Evidence = { chunk_id: string; reference_label: string; used: boolean; document_id: string; document_name: string; page_start: number | null; page_end: number | null; excerpt: string; locations: SourceLocation[] };
export type Claim = { id: string; text: string; citations: { chunk_id: string }[] };
export type GroundedAnswer = { message_id?: string; rag_run_id: string; turn_title?: string | null; answer_text: string; insufficient_evidence: boolean; claims: Claim[]; evidence: Record<string, Evidence>; retrieved_chunk_ids: string[] };
export type DocumentSummary = { document_id: string; ingestion_id: string; status: "pending" | "generating" | "ready" | "failed"; rag_run_id: string | null; error: string | null; page_count: number | null; answer?: GroundedAnswer };
export type Source = { document: { id: string; name: string; created_at: string }; ingestion: { id: string; status: string; error: string | null; created_at: string } | null; summary: DocumentSummary | null };
