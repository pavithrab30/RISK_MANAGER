import pytest

from app.core.errors import LLMProviderError
from app.llm.fallback_client import FallbackLLMClient


class _StubClient:
    def __init__(self, name: str, response: str | None = None, fails: bool = False):
        self.name = name
        self._response = response
        self._fails = fails
        self.calls = 0

    def complete(self, system, user, *, temperature=0.0):
        self.calls += 1
        if self._fails:
            raise LLMProviderError(f"{self.name} failed")
        return self._response


def test_uses_primary_when_it_succeeds():
    primary = _StubClient("primary", response="primary answer")
    fallback = _StubClient("fallback", response="fallback answer")
    client = FallbackLLMClient(primary, fallback)

    result = client.complete("sys", "user")

    assert result == "primary answer"
    assert primary.calls == 1
    assert fallback.calls == 0


def test_falls_back_when_primary_fails():
    primary = _StubClient("primary", fails=True)
    fallback = _StubClient("fallback", response="fallback answer")
    client = FallbackLLMClient(primary, fallback)

    result = client.complete("sys", "user")

    assert result == "fallback answer"
    assert primary.calls == 1
    assert fallback.calls == 1


def test_reraises_when_primary_fails_and_no_fallback_configured():
    primary = _StubClient("primary", fails=True)
    client = FallbackLLMClient(primary, None)

    with pytest.raises(LLMProviderError):
        client.complete("sys", "user")


def test_reraises_when_both_primary_and_fallback_fail():
    primary = _StubClient("primary", fails=True)
    fallback = _StubClient("fallback", fails=True)
    client = FallbackLLMClient(primary, fallback)

    with pytest.raises(LLMProviderError):
        client.complete("sys", "user")
    assert primary.calls == 1
    assert fallback.calls == 1
