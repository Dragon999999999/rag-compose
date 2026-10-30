import { request } from "./client";
import type { GroundedAnswer } from "./types";
export const ragApi = { evidence: (runId: string) => request<GroundedAnswer>(`/runs/${runId}/evidence`) };
