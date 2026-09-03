"""
Dense embedding model wrapper.

Uses a small, open-source sentence-transformers model
(BAAI/bge-small-en-v1.5 by default - ~130MB, CPU-friendly, strong MTEB score
for its size) rather than a hosted embedding API. Reasons:

  1. Cost/quota: embeddings are computed for every chunk of every ingested
     document plus every query - the highest-volume call in the system. An
     external API here means rate limits and $ cost scale with corpus size;
     a local model is free and has no quota.
  2. Determinism for evaluation: gold-set retrieval metrics need to be
     reproducible run over run. A local, versioned model gives that; a
     remote API can silently change behind the scenes.
  3. Latency: no network round-trip for the hot path of every query.

The trade-off (documented in the ADR) is lower ceiling than a top-tier
proprietary embedding model - acceptable given this is paired with BM25 +
reranking rather than relying on embedding quality alone.
"""
from __future__ import annotations

import threading

import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)

_lock = threading.Lock()
_model_cache: dict[str, object] = {}


class EmbeddingModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = self._load(model_name)

    @staticmethod
    def _load(model_name: str):
        with _lock:
            if model_name not in _model_cache:
                from sentence_transformers import SentenceTransformer

                logger.info("loading_embedding_model", model=model_name)
                _model_cache[model_name] = SentenceTransformer(model_name)
            return _model_cache[model_name]

    def encode(self, texts: list[str], *, is_query: bool = False) -> np.ndarray:
        """BGE models are trained with an instruction prefix for queries
        (not for passages) - applying it measurably improves retrieval for
        this model family, so we handle it here rather than at call sites."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if is_query and "bge" in self.model_name.lower():
            texts = [f"Represent this sentence for searching relevant passages: {t}" for t in texts]
        vectors = self._model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        return vectors.astype(np.float32)

    @property
    def dim(self) -> int:
        return self._model.get_sentence_embedding_dimension()
