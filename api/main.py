"""FastAPI backend for the LLM-Powered RAG Chatbot.

Run locally:
    uvicorn api.main:app --reload --port 8000

Then open http://localhost:8000/docs for an interactive Swagger UI, or
http://localhost:8000/redoc for the ReDoc view.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from src.config import get_settings
from src.ingest import ingest_uploaded_file, ingest_urls
from src.rag_chain import RAGChatbot, build_chat_history
from src.vector_store import count_documents, reset_collection

from .schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IngestResponse,
    IngestUrlsRequest,
    MessageResponse,
    Source,
    StatusResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("rag-api")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="LLM-Powered RAG Chatbot API",
    description=(
        "REST API for a Retrieval-Augmented Generation chatbot built on "
        "LangChain, ChromaDB, and pluggable LLM providers (Groq / OpenAI / Ollama). "
        "Use the endpoints below to ingest documents and chat with them."
    ),
    version="1.0.0",
    contact={"name": "Leela Prasad Dondla"},
    license_info={"name": "MIT"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_chatbot() -> RAGChatbot:
    """Lazy-init the RAG chatbot once per process (heavy: loads embeddings)."""
    return RAGChatbot()


# ---------------------------------------------------------------------------
# Meta endpoints
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Redirect the bare root URL to the Swagger UI."""
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    """Liveness probe used by load balancers / Docker / k8s."""
    return HealthResponse()


@app.get("/status", response_model=StatusResponse, tags=["meta"])
def status_endpoint() -> StatusResponse:
    """Inspect the current configuration and how many vectors are stored."""
    s = get_settings()
    return StatusResponse(
        vectors_count=count_documents(),
        provider=s.llm_provider,
        embedding_model=s.embedding_model,
        chunk_size=s.chunk_size,
        chunk_overlap=s.chunk_overlap,
        top_k=s.top_k,
    )


# ---------------------------------------------------------------------------
# Ingestion endpoints
# ---------------------------------------------------------------------------


@app.post(
    "/ingest/files",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["ingest"],
    summary="Upload one or more files (PDF / TXT / MD / DOCX) and add to the vector store.",
)
def ingest_files_endpoint(
    files: list[UploadFile] = File(..., description="Files to ingest."),
) -> IngestResponse:
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required.")

    total = 0
    items: list[str] = []
    for upload in files:
        try:
            data = upload.file.read()
            added = ingest_uploaded_file(upload.filename, data, DATA_DIR)
            total += added
            items.append(upload.filename)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to ingest %s", upload.filename)
            raise HTTPException(
                status_code=500, detail=f"Failed to ingest {upload.filename}: {exc}"
            ) from exc

    get_chatbot.cache_clear()
    return IngestResponse(items=items, chunks_added=total)


@app.post(
    "/ingest/urls",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["ingest"],
    summary="Fetch one or more URLs, extract text, and add to the vector store.",
)
def ingest_urls_endpoint(payload: IngestUrlsRequest) -> IngestResponse:
    urls = [str(u) for u in payload.urls]
    try:
        added = ingest_urls(urls)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to ingest URLs")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    get_chatbot.cache_clear()
    return IngestResponse(items=urls, chunks_added=added)


# ---------------------------------------------------------------------------
# Chat endpoint
# ---------------------------------------------------------------------------


@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["chat"],
    summary="Ask a question grounded in the ingested documents.",
)
def chat_endpoint(
    payload: ChatRequest,
    bot: RAGChatbot = Depends(get_chatbot),
) -> ChatResponse:
    if count_documents() == 0:
        raise HTTPException(
            status_code=409,
            detail=(
                "No documents have been ingested yet. "
                "Call POST /ingest/files or POST /ingest/urls first."
            ),
        )

    pairs: list[tuple[str, str]] = []
    msgs = payload.history
    for i in range(0, len(msgs) - 1, 2):
        if msgs[i].role == "user" and msgs[i + 1].role == "assistant":
            pairs.append((msgs[i].content, msgs[i + 1].content))

    try:
        response = bot.ask(payload.question, chat_history=build_chat_history(pairs))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Chat failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    sources = [Source(**s) for s in response.formatted_sources()]
    return ChatResponse(answer=response.answer, sources=sources)


# ---------------------------------------------------------------------------
# Admin endpoint
# ---------------------------------------------------------------------------


@app.delete(
    "/collection",
    response_model=MessageResponse,
    tags=["admin"],
    summary="Wipe every vector in the collection (irreversible).",
)
def reset_endpoint() -> MessageResponse:
    reset_collection()
    get_chatbot.cache_clear()
    return MessageResponse(message="Vector store cleared.")
