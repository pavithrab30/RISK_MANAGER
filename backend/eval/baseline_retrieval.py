"""
The "basic chunk -> embed -> cosine similarity" baseline the assignment
brief explicitly says is easy to stand up and not enough. Implemented here
as a deliberately naive counterpart to retrieval/pipeline.py, so the eval
harness can measure the actual delta the hybrid+rerank+decomposition+graph
stack buys over it, on the exact same indexed chunks.
"""
from __future__ import annotations

from app.data.models import RetrievedChunk
from app.data.store.embedding import EmbeddingModel
from app.data.store.metadata_store import MetadataStore
from app.data.store.vector_store import VectorStore


def baseline_retrieve(
    query: str,
    vector_store: VectorStore,
    metadata_store: MetadataStore,
    embedding_model: EmbeddingModel,
    top_k: int = 8,
    document_ids: list[str] | None = None,
) -> list[RetrievedChunk]:
    query_vec = embedding_model.encode([query], is_query=True)[0]
    dense_results = vector_store.query(query_vec, top_k, document_ids)
    chunk_ids = [cid for cid, _ in dense_results]
    chunk_map = metadata_store.get_chunks_by_ids(chunk_ids)

    out = []
    for chunk_id, score in dense_results:
        chunk = chunk_map.get(chunk_id)
        if chunk is None:
            continue
        out.append(RetrievedChunk(chunk=chunk, dense_score=score, fused_score=score, rerank_score=score))
    return out
