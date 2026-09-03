"""
Retrieval quality metrics computed against the gold set.

All three operate on the same input: the ranked list of (document_filename,
page_number) pairs the retriever actually returned, versus the gold set's
list of acceptable (document, page) targets for that question.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GoldTarget:
    document: str
    pages: list[int]


def _is_hit(retrieved: tuple[str, int], targets: list[GoldTarget]) -> bool:
    doc, page = retrieved
    return any(doc == t.document and page in t.pages for t in targets)


def hit_at_k(retrieved: list[tuple[str, int]], targets: list[GoldTarget], k: int) -> float:
    """1.0 if any of the top-k retrieved (document, page) pairs matches any
    gold target, else 0.0. Undefined (returns 0.0) for refusal questions
    with no targets - callers should exclude those from the aggregate."""
    if not targets:
        return 0.0
    return 1.0 if any(_is_hit(r, targets) for r in retrieved[:k]) else 0.0


def reciprocal_rank(retrieved: list[tuple[str, int]], targets: list[GoldTarget]) -> float:
    """1/rank of the first retrieved item that matches a gold target, 0 if none."""
    if not targets:
        return 0.0
    for i, r in enumerate(retrieved, start=1):
        if _is_hit(r, targets):
            return 1.0 / i
    return 0.0


def page_recall(retrieved: list[tuple[str, int]], targets: list[GoldTarget]) -> float:
    """Fraction of DISTINCT gold (document, page) targets that appear
    anywhere in the retrieved set. This is the metric that specifically
    rewards cross-page retrieval: a question with 2 gold pages needs BOTH
    to show up for a score of 1.0, not just one of them."""
    all_target_pairs = {(t.document, p) for t in targets for p in t.pages}
    if not all_target_pairs:
        return 0.0
    retrieved_set = set(retrieved)
    # a target counts as covered if the (document, page) pair, or an
    # equivalent page within the same target's accepted-pages list, is hit
    covered = 0
    for t in targets:
        if any((t.document, p) in retrieved_set for p in t.pages):
            covered += 1
    return covered / len(targets)
