import type { ApiError, DocumentSummary, ExportRequest, QueryResponse } from "./types";

const BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export class ApiRequestError extends Error {
  status: number;
  payload: ApiError;

  constructor(status: number, payload: ApiError) {
    super(payload.message);
    this.status = status;
    this.payload = payload;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, init);
  if (!res.ok) {
    const payload = (await res.json().catch(() => ({
      code: "unknown_error",
      message: `Request failed with status ${res.status}`,
      details: {},
      trace_id: "-",
    }))) as ApiError;
    throw new ApiRequestError(res.status, payload);
  }
  return res.json() as Promise<T>;
}

export async function uploadDocument(file: File): Promise<DocumentSummary> {
  const form = new FormData();
  form.append("file", file);
  const result = await request<{ document: DocumentSummary }>("/api/documents", {
    method: "POST",
    body: form,
  });
  return result.document;
}

export function listDocuments(): Promise<DocumentSummary[]> {
  return request<DocumentSummary[]>("/api/documents");
}

export function getDocument(id: string): Promise<DocumentSummary> {
  return request<DocumentSummary>(`/api/documents/${id}`);
}

export function documentFileUrl(id: string): string {
  return `${BASE_URL}/api/documents/${id}/file`;
}

export function pageImageUrl(id: string, pageNumber: number): string {
  return `${BASE_URL}/api/documents/${id}/pages/${pageNumber}/image`;
}

export function askQuestion(question: string, documentIds?: string[]): Promise<QueryResponse> {
  return request<QueryResponse>("/api/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, document_ids: documentIds ?? null }),
  });
}

async function downloadExport(path: string, payload: ExportRequest, filename: string): Promise<void> {
  const res = await fetch(`${BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const errPayload = (await res.json().catch(() => ({
      code: "unknown_error",
      message: `Export failed with status ${res.status}`,
      details: {},
      trace_id: "-",
    }))) as ApiError;
    throw new ApiRequestError(res.status, errPayload);
  }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

export function exportMarkdown(payload: ExportRequest): Promise<void> {
  return downloadExport("/api/export/markdown", payload, "docintel-answer.md");
}

export function exportPdf(payload: ExportRequest): Promise<void> {
  return downloadExport("/api/export/pdf", payload, "docintel-answer.pdf");
}
