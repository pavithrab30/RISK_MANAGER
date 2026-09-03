"""
Groq client - primary answer-generation and query-decomposition LLM.

Chosen over a local Ollama model as the *primary* generator because answer
quality directly drives the faithfulness/groundedness eval numbers, and a
free-tier hosted 70B model meaningfully outperforms what fits on a normal
laptop's CPU. Groq's free tier has no credit card requirement and very low
latency (LPU inference), which matters for query decomposition adding a
second LLM round-trip per question. Retries with backoff via tenacity so a
transient 429/5xx doesn't take down a whole request; after retries are
exhausted we raise a typed LLMProviderError so the caller can degrade
gracefully (see services/generation_service.py) instead of hanging.
"""
from __future__ import annotations

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.errors import LLMProviderError
from app.core.logging import get_logger

logger = get_logger(__name__)


class GroqClient:
    name = "groq"

    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise LLMProviderError("GROQ_API_KEY is not set - see .env.example")
        from groq import Groq

        self._client = Groq(api_key=api_key)
        self.model = model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _call(self, system: str, user: str, temperature: float) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    def complete(self, system: str, user: str, *, temperature: float = 0.0) -> str:
        try:
            return self._call(system, user, temperature)
        except Exception as exc:
            logger.error("groq_call_failed", error=str(exc))
            raise LLMProviderError(f"Groq generation failed: {exc}") from exc
