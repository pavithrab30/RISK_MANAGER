"""
Export endpoints: turn an already-answered question (the QueryResponse the
frontend already has) into a downloadable Markdown or PDF report with
citations linked back to source pages.
"""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import Response

from app.api.schemas import ExportRequest
from app.core.logging import get_logger
from app.services.export_service import generate_markdown, generate_pdf

logger = get_logger(__name__)
router = APIRouter(prefix="/api/export", tags=["export"])


@router.post("/markdown")
async def export_markdown(export: ExportRequest, request: Request) -> Response:
    content = generate_markdown(export, base_url=str(request.base_url))
    logger.info("export_markdown", question=export.question, num_citations=len(export.citations))
    return Response(
        content=content,
        media_type="text/markdown",
        headers={"Content-Disposition": 'attachment; filename="docintel-answer.md"'},
    )


@router.post("/pdf")
async def export_pdf(export: ExportRequest, request: Request) -> Response:
    content = generate_pdf(export, base_url=str(request.base_url))
    logger.info("export_pdf", question=export.question, num_citations=len(export.citations))
    return Response(
        content=content,
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="docintel-answer.pdf"'},
    )
