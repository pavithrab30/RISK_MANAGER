import pytest

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
)
from app.data.store.metadata_store import MetadataStore


@pytest.fixture
def store(tmp_path):
    return MetadataStore(tmp_path / "test.sqlite3")


def _make_chunk(document_id: str, page: int, text: str) -> Chunk:
    return Chunk.new(
        document_id=document_id,
        page_number=page,
        block_type=BlockType.TEXT,
        text=text,
        bbox=BBox(0.1, 0.1, 0.5, 0.2),
    )


def test_document_lifecycle(store):
    doc = Document.new("report.pdf", "/tmp/report.pdf")
    store.create_document(doc)

    fetched = store.get_document(doc.id)
    assert fetched.filename == "report.pdf"
    assert fetched.status == DocumentStatus.PENDING

    store.update_document_status(doc.id, DocumentStatus.READY, num_pages=5, is_scanned=True)
    fetched = store.get_document(doc.id)
    assert fetched.status == DocumentStatus.READY
    assert fetched.num_pages == 5
    assert fetched.is_scanned is True

    assert store.get_document("nonexistent") is None
    assert len(store.list_documents()) == 1


def test_pages_scoped_to_document(store):
    doc = Document.new("a.pdf", "/tmp/a.pdf")
    store.create_document(doc)
    store.add_page(Page.new(doc.id, 1, 612, 792))
    store.add_page(Page.new(doc.id, 2, 612, 792))

    pages = store.get_pages(doc.id)
    assert [p.page_number for p in pages] == [1, 2]


def test_chunk_crud_and_lookup(store):
    doc = Document.new("a.pdf", "/tmp/a.pdf")
    store.create_document(doc)
    chunk = _make_chunk(doc.id, page=3, text="Widgets have a failure rate of 2%.")
    store.add_chunk(chunk)

    fetched = store.get_chunk(chunk.id)
    assert fetched.text == chunk.text
    assert fetched.page_number == 3

    by_ids = store.get_chunks_by_ids([chunk.id, "missing_id"])
    assert list(by_ids.keys()) == [chunk.id]

    for_doc = store.get_chunks_for_document(doc.id)
    assert len(for_doc) == 1

    for_page = store.get_chunks_for_page(doc.id, 3)
    assert len(for_page) == 1
    assert store.get_chunks_for_page(doc.id, 99) == []


def test_keyword_search_finds_matching_chunk_ranked_above_irrelevant(store):
    doc = Document.new("a.pdf", "/tmp/a.pdf")
    store.create_document(doc)
    relevant = _make_chunk(doc.id, 1, "The quarterly revenue grew by twelve percent in Q3.")
    irrelevant = _make_chunk(doc.id, 2, "The office cafeteria menu changed on Tuesday.")
    store.add_chunk(relevant)
    store.add_chunk(irrelevant)

    results = store.keyword_search("quarterly revenue Q3", top_k=10)
    result_ids = [r[0] for r in results]
    assert relevant.id in result_ids
    assert result_ids[0] == relevant.id  # best match ranked first
    # higher score = more relevant, consistent with the dense-search convention
    assert results[0][1] >= (results[1][1] if len(results) > 1 else -999)


def test_keyword_search_respects_document_id_filter(store):
    doc_a = Document.new("a.pdf", "/tmp/a.pdf")
    doc_b = Document.new("b.pdf", "/tmp/b.pdf")
    store.create_document(doc_a)
    store.create_document(doc_b)
    chunk_a = _make_chunk(doc_a.id, 1, "unique_marker_token appears here")
    chunk_b = _make_chunk(doc_b.id, 1, "unique_marker_token appears here too")
    store.add_chunk(chunk_a)
    store.add_chunk(chunk_b)

    results = store.keyword_search("unique_marker_token", top_k=10, document_ids=[doc_a.id])
    assert [r[0] for r in results] == [chunk_a.id]


def test_keyword_search_empty_query_returns_empty(store):
    assert store.keyword_search("", top_k=10) == []
    assert store.keyword_search("   ", top_k=10) == []


def test_chunk_refs_and_label_resolution(store):
    doc = Document.new("a.pdf", "/tmp/a.pdf")
    store.create_document(doc)
    source = _make_chunk(doc.id, 1, "As shown in Figure 2, performance improved.")
    target = _make_chunk(doc.id, 5, "Figure 2: performance chart")
    store.add_chunk(source)
    store.add_chunk(target)
    store.add_chunk_ref(ChunkRef.new(doc.id, source.id, "Figure 2", RefType.FIGURE))

    refs = store.get_refs_for_chunks([source.id])
    assert len(refs) == 1
    assert refs[0].target_label == "Figure 2"

    resolved = store.find_chunks_by_label(doc.id, "Figure 2")
    resolved_ids = {c.id for c in resolved}
    assert target.id in resolved_ids


def test_parent_chunk_roundtrip(store):
    doc = Document.new("a.pdf", "/tmp/a.pdf")
    store.create_document(doc)
    parent = ParentChunk.new(doc.id, page_start=1, page_end=3, title="Methods", text="Full section text.")
    store.add_parent_chunk(parent)

    fetched = store.get_parent_chunk(parent.id)
    assert fetched.page_start == 1
    assert fetched.page_end == 3
    assert fetched.title == "Methods"
    assert store.get_parent_chunk("missing") is None
