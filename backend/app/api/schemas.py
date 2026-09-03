"""
Typed request/response contracts for the HTTP API edge.

Deliberately separate from app/data/models.py (the internal representation):
these are what's actually promised to callers, versioned independently of
internal storage schema changes.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class BBoxSchema(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float


class DocumentSummarySchema(BaseModel):
    id: str
    filename: str
    status: str
    num_pages: int
    is_scanned: bool
    created_at: float
    error_message: str | None = None


class UploadDocumentResponse(BaseModel):
    document: DocumentSummarySchema


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    document_ids: list[str] | None = Field(
        default=None, description="Restrict retrieval to these document ids; omit to search all."
    )


class CitationSchema(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    page_number: int
    bbox: BBoxSchema
    snippet: str
    row_number: int | None = None


class RetrievalDebugItem(BaseModel):
    """Exposed so the UI / eval harness can show *why* a passage was
    retrieved - which stage(s) surfaced it - without re-deriving it."""

    chunk_id: str
    document_id: str
    page_number: int
    dense_score: float
    keyword_score: float
    fused_score: float
    rerank_score: float | None
    via_graph_expansion: bool
    via_subquery: str | None


class QueryResponse(BaseModel):
    trace_id: str
    answer: str
    citations: list[CitationSchema]
    refused: bool
    refusal_reason: str | None
    groundedness_coverage: float
    sub_queries: list[str]
    retrieval_debug: list[RetrievalDebugItem]


class ExportRequest(BaseModel):
    """Shape sent by the frontend to export an already-answered question as a
    standalone report - it's the QueryResponse the UI already has in hand,
    not re-derived server-side, so the export matches exactly what the user
    saw on screen."""

    question: str
    answer: str
    citations: list[CitationSchema]
    groundedness_coverage: float
    refused: bool
    sub_queries: list[str] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    code: str
    message: str
    details: dict = Field(default_factory=dict)
    trace_id: str
