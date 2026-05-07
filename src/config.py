"""Application configuration loaded from environment variables."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralised, type-safe application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- LLM provider ----
    llm_provider: Literal["groq", "openai", "ollama"] = Field(
        default="groq", description="Which LLM backend to use."
    )

    groq_api_key: str | None = None
    groq_model: str = "llama-3.3-70b-versatile"

    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    # ---- Embeddings ----
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ---- Vector store ----
    chroma_persist_dir: str = "./chroma_db"
    collection_name: str = "rag_documents"

    # ---- RAG hyperparameters ----
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 4
    temperature: float = 0.2
    max_tokens: int = 1024

    @property
    def chroma_path(self) -> Path:
        path = Path(self.chroma_persist_dir).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return a cached `Settings` instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
