"""
Cross-encoder reranker.

RRF fusion gives a good candidate *set* cheaply, but bi-encoder dense
similarity and BM25 are both weak at judging fine-grained relevance of a
specific passage to a specific question. A cross-encoder that jointly
attends over (query, passage) is much stronger at that judgment - and it's
this reranked ordering, applied across the *union* of candidates pulled from
every page and every decomposed sub-query, that lets a genuinely cross-page
answer assemble in the final top-k (see retrieval/pipeline.py).

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 - small (~80MB), CPU-friendly,
well-established for passage reranking. BAAI/bge-reranker-base is a stronger
but ~10x larger alternative, left as a config swap (RERANKER_MODEL) for
anyone running on a beefier machine.
"""
from __future__ import annotations

import threading

from app.core.logging import get_logger

logger = get_logger(__name__)

_lock = threading.Lock()
_model_cache: dict[str, object] = {}


class Reranker:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = self._load(model_name)

    @staticmethod
    def _load(model_name: str):
        with _lock:
            if model_name not in _model_cache:
                from sentence_transformers import CrossEncoder

                logger.info("loading_reranker_model", model=model_name)
                _model_cache[model_name] = CrossEncoder(model_name)
            return _model_cache[model_name]

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        pairs = [(query, p) for p in passages]
        scores = self._model.predict(pairs)
        return [float(s) for s in scores]
