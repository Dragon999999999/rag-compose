const baseUrl = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000/api";

export class ApiRequestError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiRequestError";
    this.status = status;
  }
}

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${baseUrl}${path}`, init);
  if (!response.ok) {
    const detail = await response.text();
    let message = detail || `Request failed (${response.status}).`;
    try {
      const payload = JSON.parse(detail) as {
        detail?: { message?: string } | string;
        error?: { message?: string };
      };
      message =
        (typeof payload.detail === "object" ? payload.detail.message : payload.detail) ??
        payload.error?.message ??
        message;
    } catch {
      // Preserve the raw response when the server did not return JSON.
    }
    throw new ApiRequestError(message, response.status);
  }
  return response.status === 204 ? undefined as T : response.json() as Promise<T>;
}

export function pdfUrl(documentId: string): string { return `${baseUrl}/documents/${documentId}/pdf`; }
