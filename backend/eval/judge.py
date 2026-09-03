"""
LLM-as-judge for answer faithfulness / groundedness.

Deliberately uses Gemini (google-genai) while the generator uses Groq/Llama
- a different model family judging the answer avoids the well-documented
self-preference bias a model has when grading its own output. This is a
lighter-weight alternative to wiring in the full RAGAS framework: RAGAS's
faithfulness metric is itself an LLM-judge prompt under the hood, and given
the one-week scope, a direct, inspectable prompt here is easier to defend
line-by-line than a framework dependency. (RAGAS is noted as a drop-in
alternative in the ADR.)

The judge is asked for two things because they fail independently: an
answer can be perfectly faithful to the context while still not actually
answering the question (e.g. a well-cited non-answer), or it can seem to
answer the question while smuggling in a claim the context doesn't support.
"""
from __future__ import annotations

import json
import re

from app.core.logging import get_logger
from app.llm.gemini_client import GeminiClient

logger = get_logger(__name__)

_JUDGE_SYSTEM_PROMPT = """You are an impartial evaluator grading a RAG (retrieval-augmented \
generation) system's answer. You are given the user's question, the source passages the system \
was allowed to use, and the system's answer. Score two things:

1. faithfulness (0.0-1.0): does every factual claim in the answer trace back to something stated \
in the source passages? A score of 1.0 means fully grounded with no unsupported claims. A score \
of 0.0 means the answer is entirely unsupported by the sources (hallucinated or from outside \
knowledge). Partial credit for answers that are mostly grounded with minor unsupported additions.

2. relevance (0.0-1.0): does the answer actually address what the question asked, using the \
grounded information? A perfectly faithful but off-topic or non-responsive answer should score \
low here even if faithfulness is high.

If the answer is a refusal ("the documents do not contain enough information..."), judge whether \
that refusal was CORRECT given the sources - if the sources genuinely don't support an answer, a \
refusal should score 1.0 on both dimensions; if the sources actually did contain the answer and \
the system refused anyway, score both low.

Respond with ONLY a JSON object, no prose, no markdown fences:
{"faithfulness": <float 0-1>, "relevance": <float 0-1>, "reasoning": "<one or two sentences>"}"""


def judge_answer(
    judge_client: GeminiClient,
    question: str,
    answer: str,
    source_texts: list[str],
) -> dict:
    sources_block = "\n\n".join(f"[Source {i+1}]\n{t}" for i, t in enumerate(source_texts))
    user_prompt = (
        f"Question: {question}\n\nSource passages:\n{sources_block}\n\nSystem's answer:\n{answer}"
    )
    try:
        raw = judge_client.complete(_JUDGE_SYSTEM_PROMPT, user_prompt, temperature=0.0)
        parsed = _parse_json(raw)
        if parsed is None:
            logger.warning("judge_response_unparseable", raw=raw[:200])
            return {"faithfulness": None, "relevance": None, "reasoning": "unparseable judge response"}
        return {
            "faithfulness": float(parsed.get("faithfulness", 0.0)),
            "relevance": float(parsed.get("relevance", 0.0)),
            "reasoning": str(parsed.get("reasoning", "")),
        }
    except Exception as exc:
        logger.warning("judge_call_failed", error=str(exc))
        return {"faithfulness": None, "relevance": None, "reasoning": f"judge call failed: {exc}"}


def _parse_json(raw: str) -> dict | None:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(json)?", "", raw).rstrip("`").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
