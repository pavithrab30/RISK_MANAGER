"""
Query decomposition for multi-hop / compound questions.

Plain top-k retrieval runs ONE embedding through the index and gets back
chunks clustered around that one meaning - if a question genuinely needs
facts from two different parts of a document ("compare the accuracy in
Table 2 with the setup described in Section 3"), a single query vector is a
compromise between both and tends to retrieve neither well. Splitting into
sub-queries, retrieving for each independently, and fusing the results is a
direct, inspectable fix for that failure mode - this is the main mechanism
in this codebase for genuinely cross-page/cross-section questions (the
reference-graph expansion in graph_expansion.py is a complementary,
cheaper mechanism for explicit "see Table X" style cross-references).

Cost control: we don't decompose every query (that would double LLM calls
and latency for simple, single-fact questions). A cheap heuristic gate
decides whether a question looks compound before paying for the extra LLM
round-trip; see eval/report.md for the measured effect of this gate.
"""
from __future__ import annotations

import json
import re

from app.core.logging import get_logger
from app.llm.base import LLMClient

logger = get_logger(__name__)

_COMPOUND_HINTS = re.compile(
    r"\b(compare|comparison|versus|vs\.?|difference between|both|and how|"
    r"relationship between|across|as well as)\b",
    re.IGNORECASE,
)

_DECOMPOSE_SYSTEM_PROMPT = """You break a user's question about a document into 1-3 focused \
sub-questions that can each be answered by retrieving a single passage. \
If the question is already a single, simple fact lookup, return it unchanged as the only item. \
Respond with ONLY a JSON array of strings, no prose, no markdown fences."""


class QueryDecomposer:
    def __init__(self, llm_client: LLMClient | None):
        self._llm = llm_client

    def looks_compound(self, query: str) -> bool:
        if len(query) > 160:
            return True
        if query.count("?") > 1:
            return True
        return bool(_COMPOUND_HINTS.search(query))

    def decompose(self, query: str) -> list[str]:
        """Returns 1+ sub-queries. Falls back to [query] on any failure -
        decomposition is a quality enhancement, never a hard dependency."""
        if self._llm is None:
            return [query]
        try:
            raw = self._llm.complete(_DECOMPOSE_SYSTEM_PROMPT, query, temperature=0.0)
            sub_queries = _parse_json_array(raw)
            if not sub_queries:
                return [query]
            return sub_queries[:3]
        except Exception as exc:
            logger.warning("query_decomposition_failed", error=str(exc), query=query)
            return [query]


def _parse_json_array(raw: str) -> list[str]:
    raw = raw.strip()
    # tolerate the model wrapping output in a markdown code fence anyway
    if raw.startswith("```"):
        raw = re.sub(r"^```(json)?", "", raw).rstrip("`").strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            return []
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
    if not isinstance(parsed, list):
        return []
    return [str(item).strip() for item in parsed if str(item).strip()]
