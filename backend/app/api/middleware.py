"""
Trace-id middleware + typed error -> JSON envelope translation.

Every request gets a trace_id (from an incoming X-Trace-Id header if the
caller supplies one, else generated) set into the logging contextvar before
the route runs, and echoed back in the response header - so a question can
be followed end-to-end through logs by grepping one id, and a bug report
from the frontend can carry the exact trace_id to look up.
"""
from __future__ import annotations

import time

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.errors import AppError
from app.core.logging import get_logger, get_trace_id, set_trace_id

logger = get_logger(__name__)


class TraceIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        incoming = request.headers.get("x-trace-id")
        trace_id = set_trace_id(incoming)
        start = time.perf_counter()
        logger.info("request_start", method=request.method, path=request.url.path)
        try:
            response = await call_next(request)
        except AppError as exc:
            logger.error("request_app_error", code=exc.code, message=exc.message)
            response = JSONResponse(
                status_code=exc.http_status,
                content={**exc.to_dict(), "trace_id": trace_id},
            )
        except Exception as exc:  # genuine bug - log with stack trace, return generic 500
            logger.exception("request_unhandled_error", error=str(exc))
            response = JSONResponse(
                status_code=500,
                content={
                    "code": "internal_error",
                    "message": "An unexpected error occurred.",
                    "details": {},
                    "trace_id": trace_id,
                },
            )
        duration_ms = round((time.perf_counter() - start) * 1000, 1)
        response.headers["X-Trace-Id"] = trace_id
        logger.info(
            "request_end",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration_ms,
        )
        return response
