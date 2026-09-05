"""
End-to-end integration test: real PDF -> Docling parse -> chunk -> store ->
embed -> index -> hybrid retrieve -> rerank -> generate -> citations.

The generation LLM is faked (no network call, no API key needed, fully
reproducible in CI) - everything upstream of it (parsing, chunking, both
storage backends, hybrid search, fusion, reranking) is the real
implementation. This is deliberately the one test allowed to be slow (it
loads the Docling layout model, the embedding model, and the reranker) - see
README.md for how to skip it during fast iteration.
"""
from __future__ import annotations

import re

import pytest

from app.data.parsing.docling_parser import DoclingParser
from app.data.store.embedding import EmbeddingModel
from app.data.store.metadata_store import MetadataStore
from app.data.store.vector_store import VectorStore
from app.retrieval.graph_expansion import GraphExpander
from app.retrieval.hybrid_search import HybridSearcher
from app.retrieval.pipeline import RetrievalPipeline
from app.retrieval.query_decomposition import QueryDecomposer
from app.retrieval.reranker import Reranker
from app.services.chunking_service import ChunkingService
from app.services.generation_service import GenerationService


class FakeLLMClient:
    """Cites every numbered source it's given, one sentence per source, so
    the test exercises the real citation-extraction/groundedness-gate logic
    without depending on an external API or non-deterministic model output."""

    name = "fake"

    def complete(self, system: str, user: str, *, temperature: float = 0.0) -> str:
        source_indices = sorted(set(int(m) for m in re.findall(r"\[S(\d+)\]", user)))
        if not source_indices:
            return "The provided documents do not contain enough information to answer this."
        return " ".join(f"This is fact number {i} [S{i}]." for i in source_indices)


def _build_sample_pdf(path) -> str:
    import pymupdf

    doc = pymupdf.open()
    p1 = doc.new_page()
    p1.insert_text((72, 72), "Synthetic Chargeback Evidence Record", fontsize=18)
    p1.insert_text((72, 110), "1. Order and payment", fontsize=14)
    p1.insert_text((72, 140), "Order DEMO-ORDER-1001 was paid under transaction DEMO-TXN-9001", fontsize=10)
    p1.insert_text((72, 155), "for INR 2499.00 on 2026-08-10.", fontsize=10)
    p1.insert_text((72, 185), "This fixture is synthetic and must never be submitted as evidence.", fontsize=10)

    p2 = doc.new_page()
    p2.insert_text((72, 72), "2. Delivery record", fontsize=14)
    p2.insert_text((72, 100), "Carrier tracking SYNTHETIC-TRACK-4411 records delivery", fontsize=10)
    p2.insert_text((72, 115), "to the customer address on 2026-08-13.", fontsize=10)

    out = str(path / "sample.pdf")
    doc.save(out)
    return out


@pytest.fixture(scope="module")
def pipeline_components(tmp_path_factory):
    """Real components, built once per test module (model loading is the
    expensive part - amortize it across the tests in this file)."""
    tmp_path = tmp_path_factory.mktemp("e2e")
    metadata_store = MetadataStore(tmp_path / "meta.sqlite3")
    vector_store = VectorStore(tmp_path / "chroma")
    embedding_model = EmbeddingModel("BAAI/bge-small-en-v1.5")
    reranker = Reranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser = DoclingParser()
    chunker = ChunkingService()

    hybrid = HybridSearcher(vector_store, metadata_store, embedding_model, top_k_dense=10, top_k_keyword=10)
    decomposer = QueryDecomposer(None)  # decomposition disabled - no LLM needed for this test
    graph_expander = GraphExpander(metadata_store)
    retrieval_pipeline = RetrievalPipeline(
        metadata_store,
        hybrid,
        reranker,
        decomposer,
        graph_expander,
        query_decomposition_enabled=False,
        graph_expansion_enabled=True,
        top_k_reranked=5,
    )
    generation_service = GenerationService(FakeLLMClient(), rerank_score_floor=-10.0)

    return {
        "tmp_path": tmp_path,
        "metadata_store": metadata_store,
        "vector_store": vector_store,
        "embedding_model": embedding_model,
        "parser": parser,
        "chunker": chunker,
        "retrieval_pipeline": retrieval_pipeline,
        "generation_service": generation_service,
    }


@pytest.fixture(scope="module")
def ingested_document(pipeline_components):
    from app.data.models import Document, DocumentStatus, Page

    c = pipeline_components
    pdf_path = _build_sample_pdf(c["tmp_path"])

    document = Document.new("synthetic_chargeback_evidence.pdf", pdf_path)
    c["metadata_store"].create_document(document)

    parsed = c["parser"].parse(pdf_path)
    for raw_page in parsed.pages:
        page = Page.new(document.id, raw_page.page_number, raw_page.width_pt, raw_page.height_pt)
        c["metadata_store"].add_page(page)

    chunks, parents, refs = c["chunker"].build(document.id, parsed)
    for parent in parents:
        c["metadata_store"].add_parent_chunk(parent)
    for chunk in chunks:
        c["metadata_store"].add_chunk(chunk)
    for ref in refs:
        c["metadata_store"].add_chunk_ref(ref)

    texts = [ch.text for ch in chunks]
    embeddings = c["embedding_model"].encode(texts)
    c["vector_store"].add_chunks(
        chunk_ids=[ch.id for ch in chunks],
        embeddings=embeddings,
        metadatas=[{"document_id": ch.document_id, "page_number": ch.page_number} for ch in chunks],
        documents=texts,
    )
    c["metadata_store"].update_document_status(
        document.id, DocumentStatus.READY, num_pages=len(parsed.pages), is_scanned=parsed.is_scanned
    )
    return document, chunks


def test_ingestion_produces_citable_chunks_on_both_pages(ingested_document):
    document, chunks = ingested_document
    assert len(chunks) >= 1
    pages_seen = {c.page_number for c in chunks}
    assert pages_seen.issubset({1, 2})
    for c in chunks:
        assert c.document_id == document.id
        assert 0.0 <= c.bbox.x0 <= c.bbox.x1 <= 1.0


def test_query_retrieves_and_generates_grounded_answer_with_citations(
    pipeline_components, ingested_document
):
    document, _ = ingested_document
    c = pipeline_components

    result = c["retrieval_pipeline"].retrieve(
        "What delivery evidence exists for DEMO-ORDER-1001?", document_ids=[document.id]
    )
    assert len(result.chunks) >= 1
    for rc in result.chunks:
        assert rc.chunk.document_id == document.id
        assert rc.rerank_score is not None

    generation = c["generation_service"].generate(
        "What delivery evidence exists for DEMO-ORDER-1001?", result.chunks, {document.id: document}
    )
    assert generation.refused is False
    assert len(generation.citations) >= 1
    for citation in generation.citations:
        assert citation.document_id == document.id
        assert citation.page_number in (1, 2)
        assert 0.0 <= citation.bbox.x0 <= citation.bbox.x1 <= 1.0
        assert 0.0 <= citation.bbox.y0 <= citation.bbox.y1 <= 1.0
    assert generation.groundedness_coverage == 1.0  # FakeLLMClient cites every sentence


def test_query_restricted_to_unrelated_document_id_finds_nothing(pipeline_components, ingested_document):
    c = pipeline_components
    result = c["retrieval_pipeline"].retrieve(
        "What delivery evidence exists for DEMO-ORDER-1001?", document_ids=["doc_does_not_exist"]
    )
    assert result.chunks == []
