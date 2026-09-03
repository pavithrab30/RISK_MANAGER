"""
The cross-page retrieval pipeline - orchestrates every technique in
retrieval/ into one call. This is the "core of the task" (see ADR.md
"Retrieval & Architecture"):

  query
    -> [maybe] decompose into sub-queries              (query_decomposition.py)
    -> hybrid search (dense + BM25, RRF-fused) per sub-query  (hybrid_search.py)
    -> merge candidate pools across sub-queries
    -> reference-graph expansion (pull in "Figure 3" etc.)    (graph_expansion.py)
    -> cross-encoder rerank over the full merged pool          (reranker.py)
    -> top-k reranked chunks, each still carrying its page/bbox

Each stage is independently a small, defensible technique; the reason
they're stacked rather than picking just one is that they attack different
sub-cases of "the answer isn't just top-k similarity on the original query":
paraphrase/exact-term mismatch (hybrid), compound questions needing facts
from different pages (decomposition), explicit cross-references (graph
expansion), and weak bi-encoder precision (reranking).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from app.core.logging import get_logger
from app.data.models import RetrievedChunk
from app.data.store.metadata_store import MetadataStore
from app.retrieval.graph_expansion import GraphExpander
from app.retrieval.hybrid_search import HybridSearcher
from app.retrieval.query_decomposition import QueryDecomposer
from app.retrieval.reranker import Reranker

logger = get_logger(__name__)


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    sub_queries: list[str]  # what the original query was decomposed into (== [query] if not decomposed)


class RetrievalPipeline:
    def __init__(
        self,
        metadata_store: MetadataStore,
        hybrid_searcher: HybridSearcher,
        reranker: Reranker,
        query_decomposer: QueryDecomposer,
        graph_expander: GraphExpander,
        *,
        query_decomposition_enabled: bool = True,
        graph_expansion_enabled: bool = True,
        top_k_reranked: int = 8,
    ):
        self._metadata_store = metadata_store
        self._hybrid = hybrid_searcher
        self._reranker = reranker
        self._decomposer = query_decomposer
        self._graph_expander = graph_expander
        self.query_decomposition_enabled = query_decomposition_enabled
        self.graph_expansion_enabled = graph_expansion_enabled
        self.top_k_reranked = top_k_reranked

    def retrieve(
        self, query: str, document_ids: list[str] | None = None
    ) -> RetrievalResult:
        sub_queries = [query]
        if self.query_decomposition_enabled and self._decomposer.looks_compound(query):
            sub_queries = self._decomposer.decompose(query)
            logger.info("query_decomposed", original=query, sub_queries=sub_queries)

        merged: dict[str, RetrievedChunk] = {}
        for sub_query in sub_queries:
            results = self._hybrid.search(sub_query, document_ids)
            for chunk_id, candidate in results.items():
                if chunk_id not in merged:
                    merged[chunk_id] = candidate
                else:
                    existing = merged[chunk_id]
                    merged[chunk_id] = replace(
                        existing,
                        dense_score=max(existing.dense_score, candidate.dense_score),
                        keyword_score=max(existing.keyword_score, candidate.keyword_score),
                        fused_score=max(existing.fused_score, candidate.fused_score),
                    )

        # resolve chunk_id -> Chunk objects, dropping anything that vanished
        # (e.g. a document was deleted between indexing and query)
        chunk_map = self._metadata_store.get_chunks_by_ids(list(merged.keys()))
        for chunk_id in list(merged.keys()):
            chunk = chunk_map.get(chunk_id)
            if chunk is None:
                del merged[chunk_id]
                continue
            merged[chunk_id] = replace(merged[chunk_id], chunk=chunk)

        if self.graph_expansion_enabled and merged:
            merged = self._expand_by_document(merged)

        candidates = list(merged.values())
        if not candidates:
            return RetrievalResult(chunks=[], sub_queries=sub_queries)

        passages = [c.chunk.text for c in candidates]
        rerank_scores = self._reranker.score(query, passages)
        candidates = [
            replace(c, rerank_score=score) for c, score in zip(candidates, rerank_scores)
        ]
        candidates.sort(key=lambda c: c.rerank_score, reverse=True)
        top = candidates[: self.top_k_reranked]

        logger.info(
            "retrieval_complete",
            query=query,
            n_subqueries=len(sub_queries),
            n_candidates=len(candidates),
            n_returned=len(top),
            top_pages=[(c.chunk.document_id[:12], c.chunk.page_number) for c in top],
        )
        return RetrievalResult(chunks=top, sub_queries=sub_queries)

    def _expand_by_document(
        self, merged: dict[str, RetrievedChunk]
    ) -> dict[str, RetrievedChunk]:
        by_document: dict[str, dict[str, RetrievedChunk]] = {}
        for chunk_id, candidate in merged.items():
            by_document.setdefault(candidate.chunk.document_id, {})[chunk_id] = candidate

        expanded: dict[str, RetrievedChunk] = {}
        for document_id, group in by_document.items():
            expanded.update(self._graph_expander.expand(document_id, group))
        return expanded
