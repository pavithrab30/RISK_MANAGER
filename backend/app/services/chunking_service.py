"""
Chunking strategy: turns a parser-agnostic ParsedDocument into the
parent/child chunk structure that gets indexed.

Design (justified in ADR.md "Chunking"):

  - Child chunks (retrieval units): consecutive TEXT blocks are merged up to
    a target size budget so we don't retrieve on fragments smaller than a
    sentence, but we NEVER merge across a heading/table/figure boundary -
    those always become their own chunk, and tables never get split across
    chunks (a table without its header row is unanswerable). This is
    structure-aware chunking, not fixed-size sliding windows: it costs a bit
    more logic but avoids cutting a table row or a sentence in half, which
    directly damages both grounding and citation precision.

  - Parent chunks (generation context): built at the *section* level, using
    detected headings as boundaries, and allowed to span multiple pages. When
    a section continues from page N onto page N+1, the parent chunk for a
    child on page N includes the page N+1 continuation - this is one
    concrete mechanism for cross-page answers, independent of the retrieval
    fusion described in retrieval/. If no headings are detected (common in
    scanned docs), we fall back to page-level parents.

  - Reference edges: a regex pass finds "Figure 3", "Table 2", "Section 4.1"
    style mentions in chunk text and records them as ChunkRef rows. The
    retrieval graph-expansion step (retrieval/graph_expansion.py) resolves
    these at query time to pull in the referenced region even if it doesn't
    independently rank in the top-k.
"""
from __future__ import annotations

import re

from app.core.logging import get_logger
from app.data.models import (
    BBox,
    BlockType,
    Chunk,
    ChunkRef,
    ParentChunk,
    RefType,
)
from app.data.parsing.types import ParsedDocument, RawBlock

logger = get_logger(__name__)

_TARGET_CHUNK_CHARS = 1200  # ~300 tokens

_REF_PATTERN = re.compile(
    r"\b(Fig(?:ure)?\.?\s?\d+[a-zA-Z]?|Table\s?\d+|Section\s?\d+(?:\.\d+)*|§\s?\d+(?:\.\d+)*)",
    re.IGNORECASE,
)

# Matches a caption line's own opening ("Table 4: ...", "Figure 2. ...") -
# used to find a table/figure's caption among its immediate neighbors so it
# can be attached to the table/figure chunk itself. Found by measuring a
# real failure: Docling frequently emits a table's caption as a separate
# TEXT block adjacent to the TABLE block, not merged into it. Left
# unattached, the caption (natural-language, embeds well) gets retrieved
# instead of the table (raw cell data, embeds poorly against a question),
# and reference-graph label lookup ("Table 4") can't find the table chunk
# either, since the label text lives in the caption, not the table.
_CAPTION_START_PATTERN = re.compile(r"^\s*(Fig(?:ure)?\.?\s?\d+|Table\s?\d+)[:.]", re.IGNORECASE)


def _normalize_label(raw: str) -> tuple[str, RefType]:
    raw_low = raw.lower()
    if raw_low.startswith("fig"):
        num = re.search(r"\d+[a-zA-Z]?", raw)
        return f"Figure {num.group(0)}" if num else raw, RefType.FIGURE
    if raw_low.startswith("table"):
        num = re.search(r"\d+", raw)
        return f"Table {num.group(0)}" if num else raw, RefType.TABLE
    num = re.search(r"\d+(?:\.\d+)*", raw)
    return f"Section {num.group(0)}" if num else raw, RefType.SECTION


class ChunkingService:
    def build(self, document_id: str, parsed: ParsedDocument) -> tuple[
        list[Chunk], list[ParentChunk], list[ChunkRef]
    ]:
        chunks: list[Chunk] = []
        parents: list[ParentChunk] = []
        refs: list[ChunkRef] = []

        section_stack: list[str] = []  # current heading breadcrumb

        # --- pass 1: split the block stream into section groups on headings,
        # tracking the full heading breadcrumb (e.g. "3. Methods > 3.2 Data") ---
        section_groups: list[tuple[str, str, list[RawBlock]]] = []  # (title, breadcrumb, blocks)
        group_title = "Untitled section"
        group_breadcrumb = "Untitled section"
        group_blocks: list[RawBlock] = []
        for block in parsed.blocks:
            if block.block_type == BlockType.HEADING:
                if group_blocks:
                    section_groups.append((group_title, group_breadcrumb, group_blocks))
                group_title = block.text.strip()[:200] or "Untitled section"
                group_blocks = [block]
                section_stack = _update_stack(section_stack, block)
                group_breadcrumb = " > ".join(section_stack)
            else:
                group_blocks.append(block)
        if group_blocks:
            section_groups.append((group_title, group_breadcrumb, group_blocks))
        if not section_groups:
            section_groups = [("Untitled section", "Untitled section", parsed.blocks)]

        order_index = 0
        for title, breadcrumb, blocks in section_groups:
            if not blocks:
                continue
            page_start = blocks[0].page_number
            page_end = blocks[-1].page_number
            parent_text = "\n\n".join(b.text for b in blocks if b.text.strip())
            parent = ParentChunk.new(
                document_id=document_id,
                page_start=page_start,
                page_end=page_end,
                title=title,
                text=parent_text,
            )
            parents.append(parent)

            section_path = breadcrumb
            # --- pass 2: merge consecutive TEXT blocks within this section into
            # target-sized child chunks; TABLE/FIGURE/HEADING always stand alone ---
            buffer: list[RawBlock] = []

            def flush_buffer():
                nonlocal order_index
                if not buffer:
                    return
                text = "\n".join(b.text for b in buffer if b.text.strip())
                if not text.strip():
                    buffer.clear()
                    return
                bbox = BBox.union([b.bbox for b in buffer])
                chunk = Chunk.new(
                    document_id=document_id,
                    page_number=buffer[0].page_number,
                    block_type=BlockType.TEXT,
                    text=text,
                    bbox=bbox,
                    section_path=section_path,
                    parent_chunk_id=parent.id,
                    order_index=order_index,
                )
                order_index += 1
                chunks.append(chunk)
                for ref in _extract_refs(document_id, chunk):
                    refs.append(ref)
                buffer.clear()

            for block in blocks:
                if block.block_type == BlockType.HEADING:
                    continue  # heading text is already the section title/context
                if block.block_type in (BlockType.TABLE, BlockType.FIGURE, BlockType.CAPTION):
                    flush_buffer()
                    chunk = Chunk.new(
                        document_id=document_id,
                        page_number=block.page_number,
                        block_type=block.block_type,
                        text=block.text,
                        bbox=block.bbox,
                        section_path=section_path,
                        parent_chunk_id=parent.id,
                        order_index=order_index,
                        table_cells=block.table_cells,
                    )
                    order_index += 1
                    chunks.append(chunk)
                    for ref in _extract_refs(document_id, chunk):
                        refs.append(ref)
                    continue

                buffer.append(block)
                current_len = sum(len(b.text) for b in buffer)
                if current_len >= _TARGET_CHUNK_CHARS:
                    flush_buffer()
            flush_buffer()

        _attach_adjacent_captions(chunks)

        logger.info(
            "chunking_complete",
            document_id=document_id,
            num_chunks=len(chunks),
            num_parents=len(parents),
            num_refs=len(refs),
        )
        return chunks, parents, refs


def _attach_adjacent_captions(chunks: list[Chunk]) -> None:
    """Mutates TABLE/FIGURE chunks in place: if the immediately preceding or
    following chunk on the same page is that table/figure's caption, prepend
    the caption line to the table/figure chunk's own text. See
    _CAPTION_START_PATTERN above for why this exists."""
    for i, chunk in enumerate(chunks):
        if chunk.block_type not in (BlockType.TABLE, BlockType.FIGURE):
            continue
        for neighbor_idx in (i - 1, i + 1):
            if not (0 <= neighbor_idx < len(chunks)):
                continue
            neighbor = chunks[neighbor_idx]
            if neighbor.page_number != chunk.page_number:
                continue
            stripped = neighbor.text.strip()
            if _CAPTION_START_PATTERN.match(stripped):
                caption_line = stripped.split("\n")[0]
                if caption_line not in chunk.text:
                    chunk.text = f"{caption_line}\n{chunk.text}"
                break


def _update_stack(stack: list[str], heading: RawBlock) -> list[str]:
    level = heading.heading_level or 1
    new_stack = stack[: level - 1]
    new_stack.append(heading.text.strip())
    return new_stack


def _extract_refs(document_id: str, chunk: Chunk) -> list[ChunkRef]:
    refs = []
    seen = set()
    for match in _REF_PATTERN.finditer(chunk.text):
        label, ref_type = _normalize_label(match.group(0))
        if label in seen:
            continue
        seen.add(label)
        refs.append(ChunkRef.new(document_id, chunk.id, label, ref_type))
    return refs
