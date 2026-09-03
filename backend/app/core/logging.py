"""
Structured logging with a request-scoped trace_id.

Every log line emitted anywhere in the pipeline (parsing, retrieval,
generation) is JSON with a `trace_id` field so one question can be followed
end-to-end by grepping a single id. The trace_id is set once per request by
the FastAPI middleware (see api/middleware.py) via a contextvar, so services
never need to pass it around explicitly - they just call get_logger(__name__).
"""
from __future__ import annotations

import contextvars
import logging
import sys
import uuid

import structlog

_trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="-")


def new_trace_id() -> str:
    return uuid.uuid4().hex[:16]


def set_trace_id(trace_id: str | None = None) -> str:
    tid = trace_id or new_trace_id()
    _trace_id_var.set(tid)
    return tid


def get_trace_id() -> str:
    return _trace_id_var.get()


def _add_trace_id(logger, method_name, event_dict):
    event_dict["trace_id"] = _trace_id_var.get()
    return event_dict


def configure_logging(log_level: str = "INFO") -> None:
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper(), logging.INFO),
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            _add_trace_id,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str):
    return structlog.get_logger(name)
