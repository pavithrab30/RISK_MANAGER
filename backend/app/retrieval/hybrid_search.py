"""
Single-query hybrid search: dense (Chroma) + keyword (SQLite FTS5/BM25),
combined with Reciprocal Rank Fusion. This is the unit the pipeline calls
once per sub-query (see pipeline.py for the multi-sub-query orchestration).
"""
from __future__ import annotations

from app.core.logging import get_logger
from app.data.models import RetrievedChunk
from app.data.store.embedding import EmbeddingModel
from app.data.store.metadata_store import MetadataStore
from app.data.store.vector_store import VectorStore
from app.retrieval.fusion import reciprocal_rank_fusion

logger = get_logger(__name__)


class HybridSearcher:
    def __init__(
        self,
        vector_store: VectorStore,
        metadata_store: MetadataStore,
        embedding_model: EmbeddingModel,
        top_k_dense: int = 20,
        top_k_keyword: int = 20,
    ):
        self._vector_store = vector_store
        self._metadata_store = metadata_store
        self._embedding_model = embedding_model
        self.top_k_dense = top_k_dense
        self.top_k_keyword = top_k_keyword

    def search(
        self, query: str, document_ids: list[str] | None = None
    ) -> dict[str, RetrievedChunk]:
        query_vec = self._embedding_model.encode([query], is_query=True)[0]
        dense_results = self._vector_store.query(query_vec, self.top_k_dense, document_ids)
        keyword_results = self._metadata_store.keyword_search(
            query, self.top_k_keyword, document_ids
        )

        dense_ranking = [cid for cid, _ in dense_results]
        keyword_ranking = [cid for cid, _ in keyword_results]
        fused_scores = reciprocal_rank_fusion([dense_ranking, keyword_ranking])

        dense_map = dict(dense_results)
        keyword_map = dict(keyword_results)

        candidates: dict[str, RetrievedChunk] = {}
        for chunk_id, fused_score in fused_scores.items():
            candidates[chunk_id] = RetrievedChunk(
                chunk=None,  # filled in by the pipeline once ids are resolved
                dense_score=dense_map.get(chunk_id, 0.0),
                keyword_score=keyword_map.get(chunk_id, 0.0),
                fused_score=fused_score,
                via_subquery=query,
            )
        logger.debug(
            "hybrid_search",
            query=query,
            n_dense=len(dense_results),
            n_keyword=len(keyword_results),
            n_fused=len(candidates),
        )
        return candidates
