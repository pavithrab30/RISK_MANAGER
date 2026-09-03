"""
Docling-backed parser: PDF (native or scanned) -> ParsedDocument.

Why Docling over raw text extraction (PyMuPDF/pdfplumber) or a fully
vision-embedding pipeline (ColPali-style): Docling runs a layout model
(reading order + block classification: heading/text/table/figure/caption)
and a table-structure model (TableFormer) in one pass, with OCR
(RapidOCR) automatically engaged for scanned/image-only pages - which is
exactly the "messy real documents" coverage the brief asks for, without us
hand-rolling column detection or table parsing. See ADR.md "Ingestion" for
the full trade-off discussion.

Coordinate handling: Docling's native bbox is (l, t, r, b) in PDF point
units with a BOTTOM-LEFT origin. We convert to TOP-LEFT origin and normalize
to 0..1 immediately on ingestion (`to_top_left_origin().normalized()`) so
every bbox stored anywhere in this system uses one consistent convention
(see data/models.BBox) regardless of parser backend.

Known, deliberate gaps (stated plainly rather than silently swallowed):
  - Handwriting and non-Latin scripts are not covered by the default OCR
    model.
  - Heavily skewed/rotated scans beyond Docling's built-in deskew are not
    specially handled.
  - Table cells are cited at the table's overall bounding box, not
    individual per-cell pixel boxes - Docling's TableFormer does not expose
    per-cell geometry, only per-cell (row, col) position and text. Row/column
    identity is preserved and used for citation display, but drawing a
    highlight box for a single cell is out of scope here.
"""
from __future__ import annotations

import os

# Must be set before torch is imported anywhere in the process: Docling's
# layout model attempts torch.compile on first run, which requires an MSVC
# C++ toolchain that isn't present on a plain Windows dev machine. Eager
# mode is plenty fast for the page counts this project targets.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

from app.core.errors import DocumentParseError
from app.core.logging import get_logger
from app.data.models import BBox, BlockType, TableCell
from app.data.parsing.types import ParsedDocument, RawBlock, RawPage

logger = get_logger(__name__)

_SKIP_LABELS = {"page_header", "page_footer"}
_HEADING_LABELS = {"title", "section_header"}
_TABLE_LABELS = {"table"}
_FIGURE_LABELS = {"picture"}


class DoclingParser:
    def __init__(self):
        self._converter = None  # lazy: building it loads the layout model

    def _get_converter(self):
        if self._converter is None:
            from docling.document_converter import DocumentConverter

            logger.info("loading_docling_converter")
            self._converter = DocumentConverter()
        return self._converter

    def parse(self, file_path: str) -> ParsedDocument:
        converter = self._get_converter()
        try:
            result = converter.convert(file_path)
        except Exception as exc:
            raise DocumentParseError(f"Docling failed to parse {file_path}: {exc}") from exc

        doc = result.document

        pages: list[RawPage] = []
        any_scanned = False
        for page_no, page_item in doc.pages.items():
            is_scanned = _page_is_scanned(doc, page_no)
            any_scanned = any_scanned or is_scanned
            pages.append(
                RawPage(
                    page_number=page_no,
                    width_pt=page_item.size.width,
                    height_pt=page_item.size.height,
                    is_scanned=is_scanned,
                )
            )
        page_sizes = {p.page_number: (p.width_pt, p.height_pt) for p in pages}

        blocks: list[RawBlock] = []
        for item, _level in doc.iterate_items():
            label = str(getattr(item, "label", "")).lower()
            if label in _SKIP_LABELS:
                continue
            if not getattr(item, "prov", None):
                continue  # no page/position info - can't cite it, so skip it

            prov = item.prov[0]
            page_no = prov.page_no
            page_wh = page_sizes.get(page_no)
            if page_wh is None:
                continue
            bbox = _to_normalized_bbox(prov.bbox, page_wh[1], page_wh[0])

            if label in _TABLE_LABELS:
                block = _build_table_block(item, page_no, bbox)
            elif label in _FIGURE_LABELS:
                block = _build_figure_block(item, doc, page_no, bbox)
            elif label in _HEADING_LABELS:
                block = RawBlock(
                    page_number=page_no,
                    block_type=BlockType.HEADING,
                    text=(getattr(item, "text", "") or "").strip(),
                    bbox=bbox,
                    heading_level=getattr(item, "level", 1) or 1,
                )
            else:
                text = (getattr(item, "text", "") or "").strip()
                if not text:
                    continue
                if label == "list_item":
                    text = f"- {text}"
                block = RawBlock(
                    page_number=page_no,
                    block_type=BlockType.TEXT,
                    text=text,
                    bbox=bbox,
                )
            blocks.append(block)

        if not blocks:
            raise DocumentParseError(
                f"Docling extracted no citable content blocks from {file_path} "
                "(file may be empty, corrupted, or entirely unreadable even with OCR)."
            )

        logger.info(
            "docling_parse_complete",
            file_path=file_path,
            num_pages=len(pages),
            num_blocks=len(blocks),
            any_scanned=any_scanned,
        )
        return ParsedDocument(pages=pages, blocks=blocks, is_scanned=any_scanned)


def _to_normalized_bbox(docling_bbox, page_height: float, page_width: float) -> BBox:
    from docling_core.types.doc.base import CoordOrigin

    b = docling_bbox
    if b.coord_origin == CoordOrigin.BOTTOMLEFT:
        b = b.to_top_left_origin(page_height)
    page_size_obj = _Size(page_width, page_height)
    n = b.normalized(page_size_obj)
    return BBox(
        x0=max(0.0, min(1.0, n.l)),
        y0=max(0.0, min(1.0, n.t)),
        x1=max(0.0, min(1.0, n.r)),
        y1=max(0.0, min(1.0, n.b)),
    )


class _Size:
    """Minimal stand-in for docling_core's Size so we don't need to import
    a page's actual PageItem just to normalize a bbox."""

    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height


def _page_is_scanned(doc, page_no: int) -> bool:
    """A page is treated as scanned if none of the document's native text
    items land on it - i.e. every character on that page came from OCR
    rather than the PDF's text layer."""
    for text_item in doc.texts:
        for prov in getattr(text_item, "prov", []) or []:
            if prov.page_no == page_no:
                return False
    return True


def _build_table_block(item, page_no: int, bbox: BBox) -> RawBlock:
    cells: list[TableCell] = []
    markdown_rows: dict[int, dict[int, str]] = {}
    try:
        table_cells = item.data.table_cells
        for cell in table_cells:
            row = getattr(cell, "start_row_offset_idx", 0)
            col = getattr(cell, "start_col_offset_idx", 0)
            text = (getattr(cell, "text", "") or "").strip()
            row_span = max(1, getattr(cell, "row_span", 1) or 1)
            col_span = max(1, getattr(cell, "col_span", 1) or 1)
            # Docling's TableFormer does not expose per-cell pixel geometry,
            # only per-cell (row, col) position - so every cell shares the
            # table's overall bbox for citation purposes (see module docstring).
            cells.append(
                TableCell(row=row, col=col, text=text, bbox=bbox, row_span=row_span, col_span=col_span)
            )
            markdown_rows.setdefault(row, {})[col] = text
    except Exception as exc:
        logger.warning("table_cell_extraction_failed", error=str(exc))

    if markdown_rows:
        linearized = _cells_to_markdown(markdown_rows)
    else:
        linearized = (getattr(item, "text", "") or "[Table: structure unavailable]").strip()

    return RawBlock(
        page_number=page_no,
        block_type=BlockType.TABLE,
        text=linearized,
        bbox=bbox,
        table_cells=cells or None,
    )


def _cells_to_markdown(rows: dict[int, dict[int, str]]) -> str:
    row_indices = sorted(rows.keys())
    max_col = max((max(cols.keys()) for cols in rows.values() if cols), default=-1)
    lines = []
    for i, r in enumerate(row_indices):
        cols = rows[r]
        cells = [cols.get(c, "") for c in range(max_col + 1)]
        lines.append("| " + " | ".join(cells) + " |")
        if i == 0:
            lines.append("| " + " | ".join("---" for _ in cells) + " |")
    return "\n".join(lines)


def _build_figure_block(item, doc, page_no: int, bbox: BBox) -> RawBlock:
    caption = ""
    try:
        captions = getattr(item, "captions", None) or []
        if captions and hasattr(captions[0], "resolve"):
            resolved = captions[0].resolve(doc)
            caption = (getattr(resolved, "text", "") or "").strip()
    except Exception as exc:
        logger.debug("figure_caption_resolution_failed", error=str(exc))
    text = caption or "[Figure - no caption text extracted]"
    return RawBlock(page_number=page_no, block_type=BlockType.FIGURE, text=text, bbox=bbox)
