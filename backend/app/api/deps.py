"""
Dependency container: builds every singleton service once at app startup
(see main.py's lifespan) and exposes typed FastAPI dependency functions so
route handlers declare what they need instead of constructing it themselves.
This is the seam that keeps business logic out of route handlers - routes
only orchestrate calls into services/.
"""
from __future__ import annotations

from dataclasses import dataclass

from fastapi import Request

from app.core.config import Settings
from app.data.parsing.docling_parser import DoclingParser
from app.data.store.embedding import EmbeddingModel
from app.data.store.metadata_store import MetadataStore
from app.data.store.vector_store import VectorStore
from app.llm.base import LLMClient
from app.retrieval.graph_expansion import GraphExpander
from app.retrieval.hybrid_search import HybridSearcher
from app.retrieval.pipeline import RetrievalPipeline
from app.retrieval.query_decomposition import QueryDecomposer
from app.retrieval.reranker import Reranker
from app.services.chunking_service import ChunkingService
from app.services.generation_service import GenerationService
from app.services.ingestion_service import IngestionService


@dataclass
class Container:
    settings: Settings
    metadata_store: MetadataStore
    vector_store: VectorStore
    embedding_model: EmbeddingModel
    generation_llm: LLMClient | None
    decomposition_llm: LLMClient | None
    retrieval_pipeline: RetrievalPipeline
    generation_service: GenerationService | None
    ingestion_service: IngestionService


def build_container(settings: Settings) -> Container:
    metadata_store = MetadataStore(settings.sqlite_path)
    vector_store = VectorStore(settings.chroma_dir)
    embedding_model = EmbeddingModel(settings.embedding_model)
    reranker = Reranker(settings.reranker_model)
    parser = DoclingParser()
    chunker = ChunkingService()

    generation_llm: LLMClient | None = None
    decomposition_llm: LLMClient | None = None
    groq_client: LLMClient | None = None
    gemini_generation_client: LLMClient | None = None
    if settings.groq_api_key:
        from app.llm.groq_client import GroqClient

        groq_client = GroqClient(settings.groq_api_key, settings.groq_model)
    if settings.gemini_api_key:
        from app.llm.gemini_client import GeminiClient

        # Reused as a fallback generator (independent daily quota from Groq),
        # separate from the GeminiClient instance used purely as the eval judge.
        gemini_generation_client = GeminiClient(settings.gemini_api_key, settings.gemini_model)

    if groq_client is not None:
        from app.llm.fallback_client import FallbackLLMClient

        generation_llm = FallbackLLMClient(groq_client, gemini_generation_client)
        decomposition_llm = generation_llm  # same client for both
    elif gemini_generation_client is not None:
        generation_llm = gemini_generation_client
        decomposition_llm = gemini_generation_client

    hybrid_searcher = HybridSearcher(
        vector_store,
        metadata_store,
        embedding_model,
        top_k_dense=settings.top_k_dense,
        top_k_keyword=settings.top_k_keyword,
    )
    graph_expander = GraphExpander(metadata_store)
    decomposer = QueryDecomposer(decomposition_llm)
    retrieval_pipeline = RetrievalPipeline(
        metadata_store,
        hybrid_searcher,
        reranker,
        decomposer,
        graph_expander,
        query_decomposition_enabled=settings.query_decomposition_enabled,
        graph_expansion_enabled=settings.graph_expansion_enabled,
        top_k_reranked=settings.top_k_reranked,
    )

    generation_service = (
        GenerationService(
            generation_llm,
            min_coverage=settings.groundedness_min_coverage,
            vision_client=gemini_generation_client,  # GeminiClient has complete_with_image()
            page_image_dir=settings.storage_dir / "pages",
        )
        if generation_llm
        else None
    )

    ingestion_service = IngestionService(
        metadata_store,
        vector_store,
        embedding_model,
        parser,
        chunker,
        page_image_dir=settings.storage_dir / "pages",
    )

    return Container(
        settings=settings,
        metadata_store=metadata_store,
        vector_store=vector_store,
        embedding_model=embedding_model,
        generation_llm=generation_llm,
        decomposition_llm=decomposition_llm,
        retrieval_pipeline=retrieval_pipeline,
        generation_service=generation_service,
        ingestion_service=ingestion_service,
    )


def get_container(request: Request) -> Container:
    return request.app.state.container
