"""
Reciprocal Rank Fusion (RRF) for combining dense and keyword rankings.

RRF over raw score normalization because dense cosine similarity and BM25
scores live on different, incomparable scales - min-max normalizing either
is fragile (sensitive to outliers, and BM25's scale shifts with corpus
statistics). RRF only uses each list's *rank*, not its score, so it's scale-
free and a well-established default for hybrid search fusion.

score(d) = sum over each ranking r that contains d of  1 / (k + rank_r(d))
"""
from __future__ import annotations

_RRF_K = 60  # standard default from the original RRF paper; dampens the
             # influence of very top-ranked items so one list can't dominate


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = _RRF_K) -> dict[str, float]:
    """rankings: list of ranked id lists (best first). Returns id -> fused score,
    higher is better. IDs absent from a given ranking simply don't contribute
    from that ranking (equivalent to an infinite rank)."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
    return scores
