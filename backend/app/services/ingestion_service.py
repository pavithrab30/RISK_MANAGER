"""
Ingestion orchestration: upload -> parse -> chunk -> render page images ->
embed -> index. This is the one place that wires the parsing, chunking, and
storage layers together for a single document, so routes stay thin.

Runs as a FastAPI background task (see api/routes_documents.py) rather than
blocking the upload request - Docling parsing of a multi-page PDF can take
tens of seconds, which is well past what an HTTP client should be made to
wait synchronously for. The client polls GET /documents/{id} for status,
which is a deliberately simple stand-in for a real task queue (documented as
the first thing to change at scale - see ADR.md "Cost & Scale").
"""
from __future__ import annotations

from pathlib import Path

from app.core.errors import DocumentParseError
from app.core.logging import get_logger
from app.data.models import Document, DocumentStatus, Page
from app.data.parsing.docling_parser import DoclingParser
from app.data.store.embedding import EmbeddingModel
from app.data.store.metadata_store import MetadataStore
from app.data.store.vector_store import VectorStore
from app.services.chunking_service import ChunkingService

logger = get_logger(__name__)


class IngestionService:
    def __init__(
        self,
        metadata_store: MetadataStore,
        vector_store: VectorStore,
        embedding_model: EmbeddingModel,
        parser: DoclingParser,
        chunker: ChunkingService,
        page_image_dir: Path,
    ):
        self._metadata = metadata_store
        self._vectors = vector_store
        self._embedder = embedding_model
        self._parser = parser
        self._chunker = chunker
        self._page_image_dir = Path(page_image_dir)
        self._page_image_dir.mkdir(parents=True, exist_ok=True)

    def register_upload(self, filename: str, file_path: str) -> Document:
        document = Document.new(filename=filename, file_path=file_path)
        self._metadata.create_document(document)
        return document

    def ingest(self, document_id: str) -> None:
        """Synchronous ingestion pipeline for one already-registered document.
        Safe to call from a background task; never raises past the top level -
        failures are recorded on the Document row instead, so a bad file
        degrades gracefully rather than crashing the process or leaving a
        request hanging (typed status, no silent failure)."""
        document = self._metadata.get_document(document_id)
        if document is None:
            logger.error("ingest_document_not_found", document_id=document_id)
            return

        self._metadata.update_document_status(document_id, DocumentStatus.PARSING)
        try:
            self._render_page_images(document)
            parsed = self._parser.parse(document.file_path)

            for raw_page in parsed.pages:
                page = Page.new(
                    document_id=document_id,
                    page_number=raw_page.page_number,
                    width_pt=raw_page.width_pt,
                    height_pt=raw_page.height_pt,
                )
                page.is_scanned = raw_page.is_scanned
                page.image_path = str(
                    self._page_image_dir / document_id / f"page_{raw_page.page_number}.png"
                )
                self._metadata.add_page(page)

            chunks, parents, refs = self._chunker.build(document_id, parsed)
            for parent in parents:
                self._metadata.add_parent_chunk(parent)
            for chunk in chunks:
                self._metadata.add_chunk(chunk)
            for ref in refs:
                self._metadata.add_chunk_ref(ref)

            self._embed_and_index(document_id, chunks)

            self._metadata.update_document_status(
                document_id,
                DocumentStatus.READY,
                num_pages=len(parsed.pages),
                is_scanned=parsed.is_scanned,
            )
            logger.info(
                "ingestion_succeeded",
                document_id=document_id,
                num_pages=len(parsed.pages),
                num_chunks=len(chunks),
                is_scanned=parsed.is_scanned,
            )
        except DocumentParseError as exc:
            logger.error("ingestion_failed_parse", document_id=document_id, error=str(exc))
            self._metadata.update_document_status(
                document_id, DocumentStatus.FAILED, error_message=str(exc)
            )
        except Exception as exc:  # last-resort catch: never leave a document stuck "parsing"
            logger.exception("ingestion_failed_unexpected", document_id=document_id, error=str(exc))
            self._metadata.update_document_status(
                document_id, DocumentStatus.FAILED, error_message=f"Unexpected error: {exc}"
            )

    def _render_page_images(self, document: Document) -> None:
        """Renders each page to a PNG for the frontend viewer and the Gemini
        vision fallback. Uses PyMuPDF directly (fast, no model needed) rather
        than Docling's own page images, since we only need this for display,
        not analysis."""
        try:
            import pymupdf

            out_dir = self._page_image_dir / document.id
            out_dir.mkdir(parents=True, exist_ok=True)
            with pymupdf.open(document.file_path) as pdf:
                for i, page in enumerate(pdf, start=1):
                    pix = page.get_pixmap(dpi=150)
                    pix.save(str(out_dir / f"page_{i}.png"))
        except Exception as exc:
            # Non-fatal: citations still work without rendered images, the
            # frontend just falls back to pdf.js rendering the original file.
            logger.warning("page_image_render_failed", document_id=document.id, error=str(exc))

    def _embed_and_index(self, document_id: str, chunks) -> None:
        if not chunks:
            return
        texts = [c.text for c in chunks]
        embeddings = self._embedder.encode(texts, is_query=False)
        metadatas = [
            {
                "document_id": c.document_id,
                "page_number": c.page_number,
                "block_type": c.block_type.value,
            }
            for c in chunks
        ]
        self._vectors.add_chunks(
            chunk_ids=[c.id for c in chunks],
            embeddings=embeddings,
            metadatas=metadatas,
            documents=texts,
        )
