"""
Typed error hierarchy.

Every failure mode the pipeline can hit is a distinct exception type with a
stable `code`, rather than bare strings/generic exceptions bubbling up. The
API layer (api/middleware.py) catches AppError subclasses and turns them into
a consistent JSON error envelope; anything else is a genuine bug and is
logged with a stack trace as a 500.
"""
from __future__ import annotations


class AppError(Exception):
    """Base class for all expected, typed application errors."""

    code: str = "app_error"
    http_status: int = 500

    def __init__(self, message: str, *, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "details": self.details}


class DocumentParseError(AppError):
    """Docling (or fallback parser) could not extract structure from a file."""

    code = "document_parse_error"
    http_status = 422


class UnsupportedFileTypeError(AppError):
    code = "unsupported_file_type"
    http_status = 415


class DocumentNotFoundError(AppError):
    code = "document_not_found"
    http_status = 404


class RetrievalError(AppError):
    """Vector/keyword store failed and no degraded path succeeded either."""

    code = "retrieval_error"
    http_status = 502


class LLMProviderError(AppError):
    """Upstream LLM API failed after retries. Caller should degrade gracefully
    (e.g. fall back to a secondary provider or return retrieval-only results)
    rather than silently drop the request."""

    code = "llm_provider_error"
    http_status = 502


class GroundingRefusalError(AppError):
    """Not really an error - raised internally to signal 'the documents do not
    support this answer' so the API can return a clean, typed refusal instead
    of a hallucinated answer. Caught in the route handler."""

    code = "grounding_refusal"
    http_status = 200
