"""
Automatic provider fallback.

Both Groq and Gemini's free tiers have independent daily/per-minute quotas.
Rather than the whole app going down (or every question refusing) when one
provider's quota is exhausted - which happened during development, see
services/generation_service.py's docstring for the measured incident - a
FallbackLLMClient tries the primary first and transparently falls back to a
secondary provider on a typed LLMProviderError. This is graceful
degradation, not silent failure: the fallback is logged, and if both
providers fail the original error still propagates so the caller's own
typed error handling (API 502, eval per-item skip) still applies.
"""
from __future__ import annotations

from app.core.errors import LLMProviderError
from app.core.logging import get_logger
from app.llm.base import LLMClient

logger = get_logger(__name__)


class FallbackLLMClient:
    name = "fallback"

    def __init__(self, primary: LLMClient, fallback: LLMClient | None):
        self._primary = primary
        self._fallback = fallback

    def complete(self, system: str, user: str, *, temperature: float = 0.0) -> str:
        try:
            return self._primary.complete(system, user, temperature=temperature)
        except LLMProviderError as exc:
            if self._fallback is None:
                raise
            logger.warning(
                "llm_primary_failed_using_fallback",
                primary=self._primary.name,
                fallback=self._fallback.name,
                error=str(exc),
            )
            return self._fallback.complete(system, user, temperature=temperature)
