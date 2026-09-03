// Mirrors backend/app/api/schemas.py exactly - the typed contract at the
// frontend/backend edge. Keep in sync by hand (see README for why we didn't
// invest in an OpenAPI codegen step given the one-week scope).

export interface BBox {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

export type DocumentStatus = "pending" | "parsing" | "ready" | "failed";

export interface DocumentSummary {
  id: string;
  filename: string;
  status: DocumentStatus;
  num_pages: number;
  is_scanned: boolean;
  created_at: number;
  error_message: string | null;
}

export interface Citation {
  chunk_id: string;
  document_id: string;
  document_name: string;
  page_number: number;
  bbox: BBox;
  snippet: string;
  row_number: number | null;
}

export interface RetrievalDebugItem {
  chunk_id: string;
  document_id: string;
  page_number: number;
  dense_score: number;
  keyword_score: number;
  fused_score: number;
  rerank_score: number | null;
  via_graph_expansion: boolean;
  via_subquery: string | null;
}

export interface QueryResponse {
  trace_id: string;
  answer: string;
  citations: Citation[];
  refused: boolean;
  refusal_reason: string | null;
  groundedness_coverage: number;
  sub_queries: string[];
  retrieval_debug: RetrievalDebugItem[];
}

export interface ExportRequest {
  question: string;
  answer: string;
  citations: Citation[];
  groundedness_coverage: number;
  refused: boolean;
  sub_queries: string[];
}

export interface ApiError {
  code: string;
  message: string;
  details: Record<string, unknown>;
  trace_id: string;
}
