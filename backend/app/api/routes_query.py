"""
Query endpoint: the one place that composes retrieval + generation for a
single question and shapes the typed response, including retrieval debug
info so the UI/eval harness can show *why* each citation was surfaced.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import Container, get_container
from app.api.schemas import (
    BBoxSchema,
    CitationSchema,
    QueryRequest,
    QueryResponse,
    RetrievalDebugItem,
)
from app.core.errors import LLMProviderError
from app.core.logging import get_logger, get_trace_id
from app.services.generation_service import GenerationResult

logger = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest, container: Container = Depends(get_container)) -> QueryResponse:
    if container.generation_service is None:
        raise LLMProviderError(
            "No generation LLM is configured - set GROQ_API_KEY in the backend .env file."
        )

    result = container.retrieval_pipeline.retrieve(req.question, req.document_ids)

    documents_by_id = {}
    for rc in result.chunks:
        if rc.chunk.document_id not in documents_by_id:
            doc = container.metadata_store.get_document(rc.chunk.document_id)
            if doc:
                documents_by_id[rc.chunk.document_id] = doc

    generation: GenerationResult = container.generation_service.generate(
        req.question, result.chunks, documents_by_id
    )

    citations = [
        CitationSchema(
            chunk_id=c.chunk_id,
            document_id=c.document_id,
            document_name=c.document_name,
            page_number=c.page_number,
            bbox=BBoxSchema(**c.bbox.to_dict()),
            snippet=c.snippet,
        )
        for c in generation.citations
    ]

    retrieval_debug = [
        RetrievalDebugItem(
            chunk_id=rc.chunk.id,
            document_id=rc.chunk.document_id,
            page_number=rc.chunk.page_number,
            dense_score=rc.dense_score,
            keyword_score=rc.keyword_score,
            fused_score=rc.fused_score,
            rerank_score=rc.rerank_score,
            via_graph_expansion=rc.via_graph_expansion,
            via_subquery=rc.via_subquery,
        )
        for rc in result.chunks
    ]

    logger.info(
        "query_answered",
        question=req.question,
        refused=generation.refused,
        groundedness_coverage=generation.groundedness_coverage,
        num_citations=len(citations),
    )

    return QueryResponse(
        trace_id=get_trace_id(),
        answer=generation.answer_text,
        citations=citations,
        refused=generation.refused,
        refusal_reason=generation.refusal_reason,
        groundedness_coverage=generation.groundedness_coverage,
        sub_queries=result.sub_queries,
        retrieval_debug=retrieval_debug,
    )
