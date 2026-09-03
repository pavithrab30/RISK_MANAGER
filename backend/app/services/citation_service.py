"""
Maps a generated answer back to page/bbox citations.

The generation prompt (services/generation_service.py) requires the LLM to
tag each factual sentence with a bracketed source marker like [S1], [S2]
referencing the numbered context passages it was given. This module parses
those markers back out and resolves them to concrete Citation objects
(document, page, bbox) - this is what the frontend renders as a clickable,
highlightable region.

If the model emits a sentence with no marker, or a marker that doesn't
resolve, that sentence is flagged as unsupported rather than silently
citing something incorrect - see GenerationResult.uncited_sentence_count,
used by the groundedness gate in generation_service.py.
"""
from __future__ import annotations

import re

from app.data.models import BBox, Citation, RetrievedChunk

_MARKER_PATTERN = re.compile(r"\[S(\d+)(?:,\s*row\s*(\d+))?\]")


def _row_bbox(rc: RetrievedChunk, row_num: int) -> BBox:
    """Compute a tight horizontal slice of the table bbox for a specific
    data row. row_num is 1-indexed (matches the [row N] labels in prompts).

    Strategy:
    1. Try to get the bbox from table_cells for that row — works when
       Docling computed per-cell bboxes (structured PDFs).
    2. If all cells share the same bbox (OCR/scanned case), divide the
       table height equally across the number of data rows and return the
       slice for the requested row. This is approximate but much better
       than highlighting the entire table.
    """
    chunk = rc.chunk
    table_bbox = chunk.bbox

    if chunk.table_cells:
        # row_num is 1-indexed data row; table_cells row 0 is the header
        data_row_index = row_num  # row 0 = header, row 1 = first data row
        row_cells = [c for c in chunk.table_cells if c.row == data_row_index]
        if row_cells:
            cell_bbox = row_cells[0].bbox
            # Only use it if it's not identical to the whole table bbox
            is_distinct = (
                abs(cell_bbox.y0 - table_bbox.y0) > 0.001
                or abs(cell_bbox.y1 - table_bbox.y1) > 0.001
            )
            if is_distinct:
                return BBox.union([c.bbox for c in row_cells])

    # Fallback: divide table height equally across data rows.
    # Count only actual data rows — skip the markdown header row and
    # the separator line (|---|---|).
    lines = chunk.text.splitlines()
    data_rows = []
    past_separator = False
    header_rows = 0
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        # separator line like |---|---|
        if all(c in "|-: " for c in stripped):
            past_separator = True
            continue
        # header row comes before the separator
        if not past_separator:
            header_rows += 1
            continue
        # this is a real data row
        data_rows.append(line)

    total_rows = max(len(data_rows), 1)
    # The table bbox spans header + data rows. Estimate header height as
    # proportional to the number of header rows vs total rows in the table.
    total_row_count = header_rows + total_rows
    row_height = (table_bbox.y1 - table_bbox.y0) / total_row_count
    # Data rows start after the header rows
    data_start_y = table_bbox.y0 + header_rows * row_height

    # row_num is 1-indexed
    y0 = data_start_y + (row_num - 1) * row_height
    y1 = y0 + row_height

    return BBox(x0=table_bbox.x0, y0=y0, x1=table_bbox.x1, y1=min(y1, table_bbox.y1))


def split_sentences(text: str) -> list[str]:
    # Simple, dependency-free sentence splitter - good enough for citation
    # coverage checks; not used for anything user-facing that needs
    # linguistic precision.
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\[])", text.strip())
    return [p.strip() for p in parts if p.strip()]


def extract_citations(
    answer_text: str, numbered_sources: list[RetrievedChunk]
) -> tuple[list[Citation], int, int]:
    """Returns (citations, cited_sentence_count, total_sentence_count)."""
    index_to_chunk = {i + 1: rc for i, rc in enumerate(numbered_sources)}
    sentences = split_sentences(answer_text)

    citations: list[Citation] = []
    seen_chunk_ids: set[str] = set()
    cited_sentences = 0

    for sentence in sentences:
        matches = _MARKER_PATTERN.findall(sentence)  # list of (source_num, row_num_or_empty)
        resolved_with_rows: list[tuple[RetrievedChunk, int | None]] = []
        for source_str, row_str in matches:
            source_num = int(source_str)
            row_num = int(row_str) if row_str else None
            rc = index_to_chunk.get(source_num)
            if rc:
                resolved_with_rows.append((rc, row_num))

        if resolved_with_rows:
            cited_sentences += 1

        for rc, row_num in resolved_with_rows:
            # Use (chunk_id, row_num) as the dedup key so the same chunk can
            # appear multiple times with different row numbers
            dedup_key = (rc.chunk.id, row_num)
            if dedup_key in seen_chunk_ids:
                continue
            seen_chunk_ids.add(dedup_key)
            snippet = rc.chunk.text.strip()
            if len(snippet) > 240:
                snippet = snippet[:240].rsplit(" ", 1)[0] + "..."

            bbox = _row_bbox(rc, row_num) if row_num is not None else rc.chunk.bbox

            citations.append(
                Citation(
                    chunk_id=rc.chunk.id,
                    document_id=rc.chunk.document_id,
                    document_name="",  # filled in by the caller
                    page_number=rc.chunk.page_number,
                    bbox=bbox,
                    snippet=snippet,
                    row_number=row_num,
                )
            )
    return citations, cited_sentences, len(sentences)


def strip_markers(answer_text: str) -> str:
    """Human-readable answer text with [S1] and [S1, row N]-style markers
    removed, for display above the citation list."""
    return _MARKER_PATTERN.sub("", answer_text).replace("  ", " ").strip()
