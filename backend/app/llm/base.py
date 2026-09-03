"""Common interface every LLM provider client implements, so the generation
service and the eval harness can swap providers (or use two different ones -
generator vs. judge) without caring which SDK is underneath."""
from __future__ import annotations

from typing import Protocol


class LLMClient(Protocol):
    name: str

    def complete(self, system: str, user: str, *, temperature: float = 0.0) -> str:
        """Return the model's text completion. Raises LLMProviderError (after
        internal retries) if the provider fails."""
        ...
