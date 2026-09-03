"""
Intermediate representation produced by any parser backend (Docling today,
potentially others later) before it's turned into our persisted Chunk/
ParentChunk models by the chunking service.

Keeping this as a separate, parser-agnostic layer means the chunking
strategy (parent/child windows, reference extraction) doesn't import
anything Docling-specific, and a different parser could be swapped in
without touching chunking logic.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.data.models import BBox, BlockType, TableCell


@dataclass
class RawBlock:
    page_number: int
    block_type: BlockType
    text: str
    bbox: BBox
    heading_level: int | None = None  # set only when block_type == HEADING
    table_cells: list[TableCell] | None = None


@dataclass
class RawPage:
    page_number: int
    width_pt: float
    height_pt: float
    is_scanned: bool


@dataclass
class ParsedDocument:
    pages: list[RawPage]
    blocks: list[RawBlock]  # flat list, in reading order across the whole document
    is_scanned: bool
