from app.data.models import BBox, BlockType, TableCell
from app.data.parsing.types import ParsedDocument, RawBlock, RawPage
from app.services.chunking_service import ChunkingService

_BBOX = BBox(0.1, 0.1, 0.9, 0.2)


def _text_block(page: int, text: str) -> RawBlock:
    return RawBlock(page_number=page, block_type=BlockType.TEXT, text=text, bbox=_BBOX)


def _heading(page: int, text: str, level: int = 1) -> RawBlock:
    return RawBlock(
        page_number=page, block_type=BlockType.HEADING, text=text, bbox=_BBOX, heading_level=level
    )


def _build_doc() -> ParsedDocument:
    blocks = [
        _heading(1, "1. Introduction"),
        _text_block(1, "This section introduces the widget system."),
        _text_block(1, "See Figure 1 and Table 1 for an overview, as discussed in Section 2."),
        _heading(1, "2. Methods", level=1),
        _text_block(1, "Methods described here continue onto the next page."),
        _text_block(2, "Continuing the methods description from page 1 onto page 2."),
        RawBlock(
            page_number=2,
            block_type=BlockType.TABLE,
            text="| Metric | Value |\n| --- | --- |\n| Accuracy | 92% |",
            bbox=_BBOX,
            table_cells=[
                TableCell(row=0, col=0, text="Metric", bbox=_BBOX),
                TableCell(row=0, col=1, text="Value", bbox=_BBOX),
                TableCell(row=1, col=0, text="Accuracy", bbox=_BBOX),
                TableCell(row=1, col=1, text="92%", bbox=_BBOX),
            ],
        ),
        RawBlock(
            page_number=2, block_type=BlockType.FIGURE, text="Figure 1: widget diagram", bbox=_BBOX
        ),
    ]
    pages = [
        RawPage(page_number=1, width_pt=612, height_pt=792, is_scanned=False),
        RawPage(page_number=2, width_pt=612, height_pt=792, is_scanned=False),
    ]
    return ParsedDocument(pages=pages, blocks=blocks, is_scanned=False)


def test_tables_and_figures_always_become_their_own_chunk():
    chunks, _, _ = ChunkingService().build("doc1", _build_doc())
    table_chunks = [c for c in chunks if c.block_type == BlockType.TABLE]
    figure_chunks = [c for c in chunks if c.block_type == BlockType.FIGURE]
    assert len(table_chunks) == 1
    assert len(figure_chunks) == 1
    assert table_chunks[0].table_cells is not None
    assert len(table_chunks[0].table_cells) == 4
    assert table_chunks[0].page_number == 2


def test_table_text_is_never_merged_with_surrounding_prose():
    chunks, _, _ = ChunkingService().build("doc1", _build_doc())
    for chunk in chunks:
        if chunk.block_type == BlockType.TEXT:
            assert "Metric" not in chunk.text  # table content stayed isolated


def test_parent_chunk_spans_across_pages_when_section_does():
    _, parents, _ = ChunkingService().build("doc1", _build_doc())
    methods_parent = next(p for p in parents if p.title.startswith("2. Methods"))
    assert methods_parent.page_start == 1
    assert methods_parent.page_end == 2  # the whole point: cross-page parent context
    assert "continuing" in methods_parent.text.lower()


def test_reference_extraction_finds_figure_table_and_section_mentions():
    chunks, _, refs = ChunkingService().build("doc1", _build_doc())
    labels = {(r.target_label, r.ref_type.value) for r in refs}
    assert ("Figure 1", "figure") in labels
    assert ("Table 1", "table") in labels
    assert ("Section 2", "section") in labels
    # the ref must point at the chunk that actually contains the mention
    ref = next(r for r in refs if r.target_label == "Figure 1")
    source_chunk = next(c for c in chunks if c.id == ref.source_chunk_id)
    assert "Figure 1" in source_chunk.text


def test_every_chunk_has_a_valid_bbox_and_page_number():
    chunks, _, _ = ChunkingService().build("doc1", _build_doc())
    for c in chunks:
        assert c.page_number in (1, 2)
        assert 0.0 <= c.bbox.x0 <= c.bbox.x1 <= 1.0
        assert 0.0 <= c.bbox.y0 <= c.bbox.y1 <= 1.0


def test_section_path_breadcrumb_reflects_current_heading():
    chunks, _, _ = ChunkingService().build("doc1", _build_doc())
    intro_chunks = [c for c in chunks if "introduces the widget" in c.text]
    assert len(intro_chunks) == 1
    assert intro_chunks[0].section_path == "1. Introduction"


def _build_doc_with_separated_caption() -> ParsedDocument:
    """Mirrors a real failure found in manual testing: Docling emits a
    table's caption as a separate TEXT block adjacent to the TABLE block,
    not merged into it - e.g. the actual 'attention_is_all_you_need.pdf'
    Table 4 on page 10."""
    blocks = [
        _heading(1, "6 Results"),
        RawBlock(
            page_number=1,
            block_type=BlockType.TABLE,
            text="| Parser | F1 |\n| --- | --- |\n| Dyer et al. | 93.3 |",
            bbox=_BBOX,
            table_cells=[TableCell(row=0, col=0, text="Parser", bbox=_BBOX)],
        ),
        _text_block(1, "Table 4: parsing results on WSJ Section 23. Our model performs well."),
    ]
    pages = [RawPage(page_number=1, width_pt=612, height_pt=792, is_scanned=False)]
    return ParsedDocument(pages=pages, blocks=blocks, is_scanned=False)


def test_table_chunk_absorbs_an_adjacent_captions_label():
    chunks, _, _ = ChunkingService().build("doc1", _build_doc_with_separated_caption())
    table_chunk = next(c for c in chunks if c.block_type == BlockType.TABLE)
    # the table's own text must now contain its label, both so a dense
    # embedding of it has natural-language context, and so reference-graph
    # label resolution (which greps chunk text) can find the table by name
    assert "Table 4" in table_chunk.text
    assert "93.3" in table_chunk.text  # original table data must still be present


def test_caption_attachment_does_not_touch_unrelated_text_chunks():
    chunks, _, _ = ChunkingService().build("doc1", _build_doc_with_separated_caption())
    caption_chunk = next(c for c in chunks if c.block_type == BlockType.TEXT)
    assert caption_chunk.text.startswith("Table 4: parsing results")  # untouched, not duplicated
