from app.data.models import BBox, BlockType, Chunk, RetrievedChunk
from app.services.citation_service import extract_citations, split_sentences, strip_markers


def _rc(text: str, page: int) -> RetrievedChunk:
    chunk = Chunk.new(
        document_id="doc1",
        page_number=page,
        block_type=BlockType.TEXT,
        text=text,
        bbox=BBox(0.1, 0.1, 0.5, 0.2),
    )
    return RetrievedChunk(chunk=chunk, rerank_score=1.0)


def test_split_sentences_basic():
    sentences = split_sentences("First fact here. Second fact here! Third one?")
    assert sentences == ["First fact here.", "Second fact here!", "Third one?"]


def test_extract_citations_resolves_markers_to_correct_chunks():
    sources = [_rc("Revenue grew 12%.", page=3), _rc("Costs fell 5%.", page=7)]
    answer = "Revenue grew 12% in Q3 [S1]. Costs fell separately [S2]."
    citations, cited, total = extract_citations(answer, sources)

    assert total == 2
    assert cited == 2
    assert len(citations) == 2
    assert citations[0].page_number == 3
    assert citations[1].page_number == 7


def test_extract_citations_sentence_with_multiple_markers():
    sources = [_rc("Table data.", page=1), _rc("Summary text.", page=2)]
    answer = "This is shown in both places [S1][S2]."
    citations, cited, total = extract_citations(answer, sources)
    assert cited == 1
    assert total == 1
    assert len(citations) == 2


def test_extract_citations_sentence_without_marker_is_not_counted_as_cited():
    sources = [_rc("Some fact.", page=1)]
    answer = "This sentence has a citation [S1]. This one does not."
    citations, cited, total = extract_citations(answer, sources)
    assert total == 2
    assert cited == 1
    assert len(citations) == 1


def test_extract_citations_out_of_range_marker_is_ignored_not_crashing():
    sources = [_rc("Only source.", page=1)]
    answer = "This cites a source that does not exist [S9]."
    citations, cited, total = extract_citations(answer, sources)
    assert citations == []
    assert cited == 0
    assert total == 1


def test_extract_citations_deduplicates_same_chunk_across_sentences():
    sources = [_rc("Repeated fact.", page=1)]
    answer = "First mention [S1]. Second mention of the same thing [S1]."
    citations, cited, total = extract_citations(answer, sources)
    assert len(citations) == 1  # same chunk cited twice -> one citation entry
    assert cited == 2


def test_strip_markers_removes_markers_cleanly():
    text = "Revenue grew 12% [S1][S2]. Costs fell [S3]."
    stripped = strip_markers(text)
    assert "[S" not in stripped
    assert "Revenue grew 12%" in stripped
