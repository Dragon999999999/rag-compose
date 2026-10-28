export type PipelineKind = "ingestion" | "query";
export type ExecutionStrategy = "linear" | "branching" | "recursive";
export type BranchStart = "retriever" | "reranker" | "assessor";
export type PipelineExecutionBranch = {
  id: string;
  start_at?: BranchStart;
  retriever?: string;
  reranker?: string;
  assessors?: string[];
  parameter_overrides?: Record<string, unknown>;
};
export type AlternativeSource = {
  alternative_id: string;
  path_id: string;
  restart_stage: BranchStart;
};
export type PipelineAlternative = {
  id: string;
  source?: AlternativeSource | null;
  branches?: PipelineExecutionBranch[];
  stage_overrides?: Record<string, unknown>;
};
export type PipelineExecution = {
  strategy?: ExecutionStrategy;
  branches?: PipelineExecutionBranch[];
  join_at?: "decision_policy";
  evidence_merge?: "replace" | "union" | "rrf";
  max_rounds?: number;
  max_stage_calls?: number;
  retry_overrides?: Record<string, unknown>[];
  retry_on?: string[];
  alternatives?: PipelineAlternative[];
  max_attempts?: number;
  final_failure_policy?: "best_result" | "last_result" | "reject" | "abstain";
};
export type Project = {
  id: string;
  name: string;
  created_at: string;
  ingestion_pipeline_id?: string | null;
  ingestion_pipeline_version?: number | null;
  ingestion_pipeline_config_hash?: string | null;
  ingestion_representation_hash?: string | null;
  index_capabilities?: {
    capabilities?: string[];
    embedding_identity?: string | null;
    dimension?: number | null;
    representation_hash?: string | null;
  };
};
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
  pipeline_kinds?: Record<string, string[]>;
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
  kind: PipelineKind;
  name: string;
  version: number;
  description: string;
  is_default: boolean;
  representation?: Record<string, unknown>;
  query?: Record<string, unknown> & { execution?: PipelineExecution };
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
  kind: PipelineKind;
  conversation_count?: number;
  config?: PipelineConfig;
};
export type PipelineCompatibility = {
  pipeline: Pipeline;
  compatible: boolean;
  required_capabilities: string[];
  missing_capabilities: string[];
  reason: string | null;
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
  project_count?: number;
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
export type Source = { document: { id: string; name: string; created_at: string }; ingestion: { id: string; status: string; error: string | null; created_at: string } | null };
