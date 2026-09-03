"""
SQLite-backed metadata + keyword store.

Holds documents/pages/chunks/parent-chunks/cross-reference edges, and doubles
as the BM25 keyword index via an FTS5 virtual table - so "hybrid search"
doesn't need a second search service, just a second query against the same
DB. Chosen over a standalone search engine (Elasticsearch/OpenSearch) because
FTS5 ships inside Python's stdlib sqlite3 build, needs zero extra
infrastructure, and is more than adequate for the corpus sizes this project
targets (see ADR.md "Cost & Scale" for where this stops being true).

All access goes through this class - no other module opens the sqlite file
directly.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

from app.core.logging import get_logger
from app.data.models import (
    BBox,
    BlockType,
    Chunk,
    ChunkRef,
    Document,
    DocumentStatus,
    Page,
    ParentChunk,
    RefType,
    TableCell,
)

logger = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    file_path TEXT NOT NULL,
    status TEXT NOT NULL,
    num_pages INTEGER NOT NULL DEFAULT 0,
    is_scanned INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS pages (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_number INTEGER NOT NULL,
    width_pt REAL NOT NULL,
    height_pt REAL NOT NULL,
    is_scanned INTEGER NOT NULL DEFAULT 0,
    image_path TEXT
);
CREATE INDEX IF NOT EXISTS idx_pages_document ON pages(document_id);

CREATE TABLE IF NOT EXISTS parent_chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_start INTEGER NOT NULL,
    page_end INTEGER NOT NULL,
    title TEXT,
    text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_parent_chunks_document ON parent_chunks(document_id);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_number INTEGER NOT NULL,
    block_type TEXT NOT NULL,
    text TEXT NOT NULL,
    bbox_x0 REAL NOT NULL,
    bbox_y0 REAL NOT NULL,
    bbox_x1 REAL NOT NULL,
    bbox_y1 REAL NOT NULL,
    section_path TEXT NOT NULL DEFAULT '',
    parent_chunk_id TEXT REFERENCES parent_chunks(id) ON DELETE SET NULL,
    order_index INTEGER NOT NULL DEFAULT 0,
    table_cells_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_page ON chunks(document_id, page_number);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    text
);

CREATE TABLE IF NOT EXISTS chunk_refs (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    source_chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    target_label TEXT NOT NULL,
    ref_type TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunk_refs_document ON chunk_refs(document_id);
CREATE INDEX IF NOT EXISTS idx_chunk_refs_source ON chunk_refs(source_chunk_id);
"""


class MetadataStore:
    def __init__(self, db_path: Path | str):
        self.db_path = str(db_path)
        self._local = threading.local()
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _get_conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            self._local.conn = conn
        return conn

    @contextmanager
    def _connect(self):
        conn = self._get_conn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    # ---------------------------------------------------------------- documents
    def create_document(self, document: Document) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO documents (id, filename, file_path, status, num_pages, "
                "is_scanned, created_at, error_message) VALUES (?,?,?,?,?,?,?,?)",
                (
                    document.id,
                    document.filename,
                    document.file_path,
                    document.status.value,
                    document.num_pages,
                    int(document.is_scanned),
                    document.created_at,
                    document.error_message,
                ),
            )

    def update_document_status(
        self,
        document_id: str,
        status: DocumentStatus,
        *,
        num_pages: int | None = None,
        is_scanned: bool | None = None,
        error_message: str | None = None,
    ) -> None:
        fields, params = ["status = ?"], [status.value]
        if num_pages is not None:
            fields.append("num_pages = ?")
            params.append(num_pages)
        if is_scanned is not None:
            fields.append("is_scanned = ?")
            params.append(int(is_scanned))
        if error_message is not None:
            fields.append("error_message = ?")
            params.append(error_message)
        params.append(document_id)
        with self._connect() as conn:
            conn.execute(f"UPDATE documents SET {', '.join(fields)} WHERE id = ?", params)

    def get_document(self, document_id: str) -> Document | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
        return _row_to_document(row) if row else None

    def list_documents(self) -> list[Document]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
        return [_row_to_document(r) for r in rows]

    # --------------------------------------------------------------------- pages
    def add_page(self, page: Page) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO pages (id, document_id, page_number, width_pt, height_pt, "
                "is_scanned, image_path) VALUES (?,?,?,?,?,?,?)",
                (
                    page.id,
                    page.document_id,
                    page.page_number,
                    page.width_pt,
                    page.height_pt,
                    int(page.is_scanned),
                    page.image_path,
                ),
            )

    def get_pages(self, document_id: str) -> list[Page]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM pages WHERE document_id = ? ORDER BY page_number", (document_id,)
            ).fetchall()
        return [
            Page(
                id=r["id"],
                document_id=r["document_id"],
                page_number=r["page_number"],
                width_pt=r["width_pt"],
                height_pt=r["height_pt"],
                is_scanned=bool(r["is_scanned"]),
                image_path=r["image_path"],
            )
            for r in rows
        ]

    # ------------------------------------------------------------- parent chunks
    def add_parent_chunk(self, parent: ParentChunk) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO parent_chunks (id, document_id, page_start, page_end, title, text) "
                "VALUES (?,?,?,?,?,?)",
                (parent.id, parent.document_id, parent.page_start, parent.page_end, parent.title, parent.text),
            )

    def get_parent_chunk(self, parent_id: str) -> ParentChunk | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM parent_chunks WHERE id = ?", (parent_id,)).fetchone()
        if not row:
            return None
        return ParentChunk(
            id=row["id"],
            document_id=row["document_id"],
            page_start=row["page_start"],
            page_end=row["page_end"],
            title=row["title"] or "",
            text=row["text"],
        )

    # ------------------------------------------------------------------- chunks
    def add_chunk(self, chunk: Chunk) -> None:
        table_cells_json = None
        if chunk.table_cells:
            table_cells_json = json.dumps(
                [
                    {
                        "row": c.row,
                        "col": c.col,
                        "text": c.text,
                        "bbox": c.bbox.to_dict(),
                        "row_span": c.row_span,
                        "col_span": c.col_span,
                    }
                    for c in chunk.table_cells
                ]
            )
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO chunks (id, document_id, page_number, block_type, text, "
                "bbox_x0, bbox_y0, bbox_x1, bbox_y1, section_path, parent_chunk_id, "
                "order_index, table_cells_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    chunk.id,
                    chunk.document_id,
                    chunk.page_number,
                    chunk.block_type.value,
                    chunk.text,
                    chunk.bbox.x0,
                    chunk.bbox.y0,
                    chunk.bbox.x1,
                    chunk.bbox.y1,
                    chunk.section_path,
                    chunk.parent_chunk_id,
                    chunk.order_index,
                    table_cells_json,
                ),
            )
            conn.execute(
                "INSERT INTO chunks_fts (chunk_id, text) VALUES (?, ?)", (chunk.id, chunk.text)
            )

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
        return _row_to_chunk(row) if row else None

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> dict[str, Chunk]:
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" * len(chunk_ids))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM chunks WHERE id IN ({placeholders})", chunk_ids
            ).fetchall()
        return {r["id"]: _row_to_chunk(r) for r in rows}

    def get_chunks_for_document(self, document_id: str) -> list[Chunk]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM chunks WHERE document_id = ? ORDER BY page_number, order_index",
                (document_id,),
            ).fetchall()
        return [_row_to_chunk(r) for r in rows]

    def get_chunks_for_page(self, document_id: str, page_number: int) -> list[Chunk]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM chunks WHERE document_id = ? AND page_number = ? ORDER BY order_index",
                (document_id, page_number),
            ).fetchall()
        return [_row_to_chunk(r) for r in rows]

    def keyword_search(
        self, query: str, top_k: int = 20, document_ids: list[str] | None = None
    ) -> list[tuple[str, float]]:
        """BM25 keyword search via FTS5. Returns (chunk_id, score) with higher =
        more relevant (we negate FTS5's native bm25(), where lower/negative is
        better, so downstream fusion code has one consistent convention: higher
        score always means better match, for both dense and keyword results).
        """
        safe_query = _sanitize_fts_query(query)
        if not safe_query:
            return []
        doc_filter = ""
        params: list = [safe_query]
        if document_ids:
            placeholders = ",".join("?" * len(document_ids))
            doc_filter = f"AND c.document_id IN ({placeholders})"
            params.extend(document_ids)
        params.append(top_k)
        sql = f"""
            SELECT f.chunk_id AS chunk_id, -bm25(chunks_fts) AS score
            FROM chunks_fts f
            JOIN chunks c ON c.id = f.chunk_id
            WHERE chunks_fts MATCH ? {doc_filter}
            ORDER BY bm25(chunks_fts) ASC
            LIMIT ?
        """
        try:
            with self._connect() as conn:
                rows = conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError as exc:
            logger.warning("keyword_search_failed", query=query, error=str(exc))
            return []
        return [(r["chunk_id"], float(r["score"])) for r in rows]

    # -------------------------------------------------------------- chunk refs
    def add_chunk_ref(self, ref: ChunkRef) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO chunk_refs (id, document_id, source_chunk_id, target_label, ref_type) "
                "VALUES (?,?,?,?,?)",
                (ref.id, ref.document_id, ref.source_chunk_id, ref.target_label, ref.ref_type.value),
            )

    def get_refs_for_chunks(self, chunk_ids: list[str]) -> list[ChunkRef]:
        if not chunk_ids:
            return []
        placeholders = ",".join("?" * len(chunk_ids))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM chunk_refs WHERE source_chunk_id IN ({placeholders})", chunk_ids
            ).fetchall()
        return [
            ChunkRef(
                id=r["id"],
                document_id=r["document_id"],
                source_chunk_id=r["source_chunk_id"],
                target_label=r["target_label"],
                ref_type=RefType(r["ref_type"]),
            )
            for r in rows
        ]

    def find_chunks_by_label(self, document_id: str, label: str) -> list[Chunk]:
        """Resolve a reference like 'Figure 3' to the chunk(s) whose caption/
        section_path mentions it - used by graph expansion."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM chunks WHERE document_id = ? AND "
                "(text LIKE ? OR section_path LIKE ?) LIMIT 5",
                (document_id, f"%{label}%", f"%{label}%"),
            ).fetchall()
        return [_row_to_chunk(r) for r in rows]


def _sanitize_fts_query(query: str) -> str:
    """FTS5 query syntax treats punctuation specially; for a natural-language
    question we just want an OR-of-terms match, so strip FTS operators and
    quote each token."""
    tokens = [t.strip('"') for t in query.replace('"', " ").split() if t.strip()]
    tokens = [t for t in tokens if t.isalnum() or any(ch.isalnum() for ch in t)]
    if not tokens:
        return ""
    cleaned = []
    for t in tokens:
        t = "".join(ch for ch in t if ch.isalnum() or ch in "-_")
        if t:
            cleaned.append(f'"{t}"')
    return " OR ".join(cleaned)


def _row_to_document(row: sqlite3.Row) -> Document:
    return Document(
        id=row["id"],
        filename=row["filename"],
        file_path=row["file_path"],
        status=DocumentStatus(row["status"]),
        num_pages=row["num_pages"],
        is_scanned=bool(row["is_scanned"]),
        created_at=row["created_at"],
        error_message=row["error_message"],
    )


def _row_to_chunk(row: sqlite3.Row) -> Chunk:
    table_cells = None
    if row["table_cells_json"]:
        raw = json.loads(row["table_cells_json"])
        table_cells = [
            TableCell(
                row=c["row"],
                col=c["col"],
                text=c["text"],
                bbox=BBox(**c["bbox"]),
                row_span=c.get("row_span", 1),
                col_span=c.get("col_span", 1),
            )
            for c in raw
        ]
    return Chunk(
        id=row["id"],
        document_id=row["document_id"],
        page_number=row["page_number"],
        block_type=BlockType(row["block_type"]),
        text=row["text"],
        bbox=BBox(row["bbox_x0"], row["bbox_y0"], row["bbox_x1"], row["bbox_y1"]),
        section_path=row["section_path"] or "",
        parent_chunk_id=row["parent_chunk_id"],
        order_index=row["order_index"],
        table_cells=table_cells,
    )
