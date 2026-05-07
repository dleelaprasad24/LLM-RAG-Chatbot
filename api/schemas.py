"""Pydantic request and response models for the FastAPI backend.

These schemas drive both runtime validation and the auto-generated OpenAPI /
Swagger docs at ``/docs``.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, HttpUrl


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    """A single turn in the conversation history (OpenAI-style)."""

    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user's question.")
    history: list[ChatMessage] = Field(
        default_factory=list,
        description="Prior turns. Used for history-aware retrieval.",
    )


class Source(BaseModel):
    source: str = Field(..., description="Filename or URL the chunk came from.")
    page: Optional[int] = Field(default=None, description="Zero-indexed page number for PDFs.")
    title: Optional[str] = Field(default=None, description="Document title or filename.")
    snippet: str = Field(..., description="First 300 chars of the supporting passage.")


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


class IngestUrlsRequest(BaseModel):
    urls: list[HttpUrl] = Field(..., min_length=1, description="URLs to fetch and ingest.")


class IngestResponse(BaseModel):
    items: list[str] = Field(..., description="Filenames or URLs that were ingested.")
    chunks_added: int


# ---------------------------------------------------------------------------
# Status / health
# ---------------------------------------------------------------------------


class StatusResponse(BaseModel):
    vectors_count: int
    provider: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class MessageResponse(BaseModel):
    message: str
