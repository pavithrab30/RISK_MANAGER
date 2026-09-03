"""
FastAPI application entrypoint. Wires the dependency container, structured
logging, trace-id middleware, and typed error handling; route modules stay
thin (see api/routes_documents.py, api/routes_query.py).
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

# See app/data/parsing/docling_parser.py for why this must be set before any
# torch import happens anywhere in the process.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.deps import build_container
from app.api.middleware import TraceIdMiddleware
from app.api.routes_documents import router as documents_router
from app.api.routes_export import router as export_router
from app.api.routes_query import router as query_router
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    logger.info("app_starting", env=settings.env)
    app.state.container = build_container(settings)
    logger.info("app_ready")
    yield
    logger.info("app_shutdown")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="DocIntel - Multimodal Document Intelligence", lifespan=lifespan)

    app.add_middleware(TraceIdMiddleware)

    # In development, allow any localhost port (Vite may pick 5173, 5174, etc.)
    # In production, restrict to the explicit CORS_ORIGINS list.
    if settings.env == "development":
        cors_origins = ["*"]
    else:
        cors_origins = settings.cors_origins_list

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=False,  # must be False when allow_origins=["*"]
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(documents_router)
    app.include_router(query_router)
    app.include_router(export_router)

    @app.get("/api/health")
    async def health():
        return {"status": "ok"}

    return app


app = create_app()
