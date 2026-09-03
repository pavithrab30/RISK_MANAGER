"""
Document upload/listing/retrieval endpoints. Thin by design: all real work
happens in services/ingestion_service.py - this module only validates the
request, delegates, and shapes the typed response.
"""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, UploadFile
from fastapi.responses import FileResponse

from app.api.deps import Container, get_container
from app.api.schemas import DocumentSummarySchema, UploadDocumentResponse
from app.core.errors import DocumentNotFoundError, UnsupportedFileTypeError
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/documents", tags=["documents"])

_ALLOWED_EXTENSIONS = {".pdf"}


def _to_summary(document) -> DocumentSummarySchema:
    return DocumentSummarySchema(
        id=document.id,
        filename=document.filename,
        status=document.status.value,
        num_pages=document.num_pages,
        is_scanned=document.is_scanned,
        created_at=document.created_at,
        error_message=document.error_message,
    )


@router.post("", response_model=UploadDocumentResponse)
async def upload_document(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    container: Container = Depends(get_container),
) -> UploadDocumentResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{suffix}'. Only PDF is supported.",
            details={"filename": file.filename},
        )

    upload_dir = container.settings.upload_dir
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = f"{uuid.uuid4().hex[:12]}_{Path(file.filename).name}"
    dest_path = upload_dir / safe_name
    with dest_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    document = container.ingestion_service.register_upload(
        filename=file.filename or safe_name, file_path=str(dest_path)
    )
    background_tasks.add_task(container.ingestion_service.ingest, document.id)

    logger.info("document_uploaded", document_id=document.id, filename=document.filename)
    return UploadDocumentResponse(document=_to_summary(document))


@router.get("", response_model=list[DocumentSummarySchema])
async def list_documents(container: Container = Depends(get_container)):
    docs = container.metadata_store.list_documents()
    result = []
    for d in docs:
        summary = _to_summary(d)
        # If the file no longer exists on disk, surface it as failed so the
        # UI doesn't show "Ready" for a document that can't actually be served.
        if d.status.value == "ready" and not Path(d.file_path).exists():
            summary = DocumentSummarySchema(
                id=d.id,
                filename=d.filename,
                status="failed",
                num_pages=d.num_pages,
                is_scanned=d.is_scanned,
                created_at=d.created_at,
                error_message="Source file missing from disk — please re-upload.",
            )
            logger.warning("document_file_missing_on_disk", document_id=d.id, path=d.file_path)
        result.append(summary)
    return result


@router.get("/{document_id}", response_model=DocumentSummarySchema)
async def get_document(document_id: str, container: Container = Depends(get_container)):
    document = container.metadata_store.get_document(document_id)
    if document is None:
        raise DocumentNotFoundError(f"Document {document_id} not found")
    return _to_summary(document)


@router.get("/{document_id}/file")
async def get_document_file(document_id: str, container: Container = Depends(get_container)):
    document = container.metadata_store.get_document(document_id)
    if document is None:
        raise DocumentNotFoundError(f"Document {document_id} not found")
    if not Path(document.file_path).exists():
        raise DocumentNotFoundError(
            f"File for document {document_id} is missing from disk — please re-upload.",
            details={"file_path": document.file_path},
        )
    return FileResponse(document.file_path, media_type="application/pdf", filename=document.filename)


@router.get("/{document_id}/pages/{page_number}/image")
async def get_page_image(
    document_id: str, page_number: int, container: Container = Depends(get_container)
):
    pages = container.metadata_store.get_pages(document_id)
    match = next((p for p in pages if p.page_number == page_number), None)
    if match is None or not match.image_path or not Path(match.image_path).exists():
        raise DocumentNotFoundError(
            f"No rendered image for document {document_id} page {page_number}"
        )
    return FileResponse(match.image_path, media_type="image/png")
