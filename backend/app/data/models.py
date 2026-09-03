"""
Internal data-layer models.

These are plain dataclasses (not the API's Pydantic schemas - see
api/schemas.py) because they're the shape data takes *inside* the system,
persisted to SQLite/Chroma. Keeping them separate from the API contracts
means we can change internal representation without breaking the public API,
and vice versa.

Everything that can eventually be cited carries page_number + bbox, because
region-level citation is a first-class requirement, not something bolted on
after generation.
"""
from __future__ import annotations

import enum
import time
import uuid
from dataclasses import dataclass, field


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class DocumentStatus(str, enum.Enum):
    PENDING = "pending"
    PARSING = "parsing"
    READY = "ready"
    FAILED = "failed"


class BlockType(str, enum.Enum):
    TEXT = "text"
    HEADING = "heading"
    TABLE = "table"
    TABLE_CELL = "table_cell"
    FIGURE = "figure"
    CAPTION = "caption"
    LIST = "list"


@dataclass(frozen=True)
class BBox:
    """Bounding box in normalized page coordinates (0..1), origin top-left.

    Normalized so the frontend can draw a highlight at any render resolution
    without needing the original PDF's point size.
    """

    x0: float
    y0: float
    x1: float
    y1: float

    def to_dict(self) -> dict:
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1}

    @staticmethod
    def union(boxes: list["BBox"]) -> "BBox":
        return BBox(
            x0=min(b.x0 for b in boxes),
            y0=min(b.y0 for b in boxes),
            x1=max(b.x1 for b in boxes),
            y1=max(b.y1 for b in boxes),
        )


@dataclass
class Document:
    id: str
    filename: str
    file_path: str
    status: DocumentStatus = DocumentStatus.PENDING
    num_pages: int = 0
    is_scanned: bool = False  # true if OCR was used for any page
    created_at: float = field(default_factory=time.time)
    error_message: str | None = None

    @staticmethod
    def new(filename: str, file_path: str) -> "Document":
        return Document(id=_new_id("doc"), filename=filename, file_path=file_path)


@dataclass
class Page:
    id: str
    document_id: str
    page_number: int  # 1-indexed
    width_pt: float
    height_pt: float
    is_scanned: bool = False
    image_path: str | None = None  # rendered PNG for the viewer / vision fallback

    @staticmethod
    def new(document_id: str, page_number: int, width_pt: float, height_pt: float) -> "Page":
        return Page(
            id=_new_id("page"),
            document_id=document_id,
            page_number=page_number,
            width_pt=width_pt,
            height_pt=height_pt,
        )


@dataclass
class TableCell:
    row: int
    col: int
    text: str
    bbox: BBox
    row_span: int = 1
    col_span: int = 1


@dataclass
class Chunk:
    """A child (retrievable) unit: one paragraph, one table, one figure caption,
    etc. This is what gets embedded and what a citation ultimately points at.
    """

    id: str
    document_id: str
    page_number: int
    block_type: BlockType
    text: str  # embeddable text - for tables, a linearized markdown rendering
    bbox: BBox
    section_path: str = ""  # e.g. "3. Methods > 3.2 Data Collection" for context
    parent_chunk_id: str | None = None  # points at the page/section-level ParentChunk
    table_cells: list[TableCell] | None = None  # populated only for block_type=TABLE
    order_index: int = 0  # reading order within the page, for windowed context

    @staticmethod
    def new(
        document_id: str,
        page_number: int,
        block_type: BlockType,
        text: str,
        bbox: BBox,
        **kwargs,
    ) -> "Chunk":
        return Chunk(
            id=_new_id("chunk"),
            document_id=document_id,
            page_number=page_number,
            block_type=block_type,
            text=text,
            bbox=bbox,
            **kwargs,
        )


@dataclass
class ParentChunk:
    """A section- or page-level window of context. When a child Chunk is
    retrieved, its ParentChunk is what actually gets fed to the LLM, so the
    model reasons over coherent context rather than an isolated fragment.
    """

    id: str
    document_id: str
    page_start: int
    page_end: int
    title: str
    text: str

    @staticmethod
    def new(document_id: str, page_start: int, page_end: int, title: str, text: str) -> "ParentChunk":
        return ParentChunk(
            id=_new_id("parent"),
            document_id=document_id,
            page_start=page_start,
            page_end=page_end,
            title=title,
            text=text,
        )


class RefType(str, enum.Enum):
    FIGURE = "figure"
    TABLE = "table"
    SECTION = "section"


@dataclass
class ChunkRef:
    """A cross-reference edge: chunk `source_chunk_id` textually references
    another region (e.g. "as shown in Figure 3", "see Table 2"). Used by the
    retrieval graph-expansion step to pull in the referenced region even when
    it wouldn't otherwise rank in the top-k. This is the lightweight
    'document graph' - edges are derived from explicit textual references and
    shared figure/table/section labels, not a general entity graph.
    """

    id: str
    document_id: str
    source_chunk_id: str
    target_label: str  # e.g. "Figure 3", "Table 2", "Section 4.1" (resolved at query time)
    ref_type: RefType

    @staticmethod
    def new(document_id: str, source_chunk_id: str, target_label: str, ref_type: RefType) -> "ChunkRef":
        return ChunkRef(
            id=_new_id("ref"),
            document_id=document_id,
            source_chunk_id=source_chunk_id,
            target_label=target_label,
            ref_type=ref_type,
        )


@dataclass
class Citation:
    chunk_id: str
    document_id: str
    document_name: str
    page_number: int
    bbox: BBox
    snippet: str  # short excerpt actually used, for display
    row_number: int | None = None  # set for table citations, points to the specific data row


@dataclass
class RetrievedChunk:
    """A Chunk plus retrieval-time scoring metadata, threaded through the
    hybrid search -> rerank -> graph-expansion -> generation pipeline so we
    can log/debug/evaluate at every stage.
    """

    chunk: Chunk
    dense_score: float = 0.0
    keyword_score: float = 0.0
    fused_score: float = 0.0
    rerank_score: float | None = None
    via_graph_expansion: bool = False
    via_subquery: str | None = None  # which decomposed sub-query surfaced this
