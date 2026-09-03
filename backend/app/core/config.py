"""
Centralized application settings.

All configuration is sourced from environment variables (or a local .env file
loaded by pydantic-settings) - nothing here is hardcoded. This is the single
place the rest of the codebase reads config from; services must not read
os.environ directly.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    env: str = Field(default="development", alias="ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # --- LLM providers ---
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.3-70b-versatile", alias="GROQ_MODEL")

    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-1.5-flash", alias="GEMINI_MODEL")

    # --- Storage ---
    storage_dir: Path = Field(default=Path("./storage"), alias="STORAGE_DIR")
    chroma_dir: Path = Field(default=Path("./storage/chroma"), alias="CHROMA_DIR")
    sqlite_path: Path = Field(default=Path("./storage/docintel.sqlite3"), alias="SQLITE_PATH")
    upload_dir: Path = Field(default=Path("./storage/uploads"), alias="UPLOAD_DIR")

    # --- Retrieval / models ---
    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5", alias="EMBEDDING_MODEL")
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2", alias="RERANKER_MODEL"
    )
    top_k_dense: int = Field(default=20, alias="TOP_K_DENSE")
    top_k_keyword: int = Field(default=20, alias="TOP_K_KEYWORD")
    top_k_reranked: int = Field(default=8, alias="TOP_K_RERANKED")
    graph_expansion_enabled: bool = Field(default=True, alias="GRAPH_EXPANSION_ENABLED")
    query_decomposition_enabled: bool = Field(default=True, alias="QUERY_DECOMPOSITION_ENABLED")

    # Groundedness gate: minimum fraction of answer sentences that must map to
    # a cited chunk before we return the answer as-is (else we degrade to a
    # partial/"not found" response). See services/generation_service.py.
    groundedness_min_coverage: float = Field(default=0.6, alias="GROUNDEDNESS_MIN_COVERAGE")

    cors_origins: str = Field(default="http://localhost:5173", alias="CORS_ORIGINS")

    def resolve_paths(self) -> None:
        """Make every storage path absolute, resolved against the process's
        CWD at startup. Without this, paths persisted to the database (e.g.
        Document.file_path) are relative strings that silently break if the
        process is ever later started from a different working directory -
        a real bug we hit during manual testing (renaming the storage
        directory broke file lookups for already-ingested documents even
        though nothing about the documents themselves had changed)."""
        self.storage_dir = self.storage_dir.resolve()
        self.chroma_dir = self.chroma_dir.resolve()
        self.sqlite_path = self.sqlite_path.resolve()
        self.upload_dir = self.upload_dir.resolve()

    def ensure_dirs(self) -> None:
        for d in (self.storage_dir, self.chroma_dir, self.upload_dir, self.sqlite_path.parent):
            d.mkdir(parents=True, exist_ok=True)

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.resolve_paths()
    settings.ensure_dirs()
    return settings
