"""
Dense vector store, backed by Chroma's local persistent client.

Chosen over standing up Qdrant/Milvus/Weaviate because Chroma runs embedded
(no separate server process, no extra docker service) while still persisting
to disk and supporting metadata filtering - which keeps the whole system a
genuine one-command run on a laptop. The trade-off (no built-in horizontal
scaling, no native hybrid search) is why keyword search is handled
separately via SQLite FTS5 and fused in the retrieval layer instead of
relying on Chroma for it. See ADR.md "Cost & Scale" for what changes at
10k+ documents.
"""
from __future__ import annotations

from pathlib import Path

from app.core.errors import RetrievalError
from app.core.logging import get_logger

logger = get_logger(__name__)

_COLLECTION_NAME = "chunks"


class VectorStore:
    def __init__(self, persist_dir: Path | str):
        import chromadb

        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(
        self,
        chunk_ids: list[str],
        embeddings,
        metadatas: list[dict],
        documents: list[str],
    ) -> None:
        if not chunk_ids:
            return
        try:
            self._collection.add(
                ids=chunk_ids,
                embeddings=embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings,
                metadatas=metadatas,
                documents=documents,
            )
        except Exception as exc:
            raise RetrievalError(f"Failed to index {len(chunk_ids)} chunk embeddings: {exc}") from exc

    def query(
        self, query_embedding, top_k: int = 20, document_ids: list[str] | None = None
    ) -> list[tuple[str, float]]:
        """Returns (chunk_id, similarity_score) sorted best-first, score in
        [-1, 1] (cosine similarity = 1 - chroma distance), consistent with the
        keyword store's "higher is better" convention used by RRF fusion."""
        where = {"document_id": {"$in": document_ids}} if document_ids else None
        vec = query_embedding.tolist() if hasattr(query_embedding, "tolist") else query_embedding
        try:
            result = self._collection.query(
                query_embeddings=[vec],
                n_results=top_k,
                where=where,
                include=["distances"],
            )
        except Exception as exc:
            logger.error("vector_query_failed", error=str(exc))
            raise RetrievalError(f"Vector search failed: {exc}") from exc

        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        return [(cid, 1.0 - dist) for cid, dist in zip(ids, distances)]

    def delete_document(self, document_id: str) -> None:
        self._collection.delete(where={"document_id": document_id})

    def count(self) -> int:
        return self._collection.count()
