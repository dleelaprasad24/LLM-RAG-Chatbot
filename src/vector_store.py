"""Persistent Chroma vector store wrapper."""

from __future__ import annotations

import logging
from typing import Iterable

from langchain_chroma import Chroma
from langchain_core.documents import Document

from .config import get_settings
from .embeddings import get_embeddings

logger = logging.getLogger(__name__)


def get_vector_store() -> Chroma:
    """Return a persistent Chroma collection."""
    settings = get_settings()
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(settings.chroma_path),
    )


def add_documents(documents: Iterable[Document]) -> int:
    """Embed and persist documents. Returns the number of chunks added."""
    docs = list(documents)
    if not docs:
        logger.warning("No documents to add.")
        return 0
    store = get_vector_store()
    store.add_documents(docs)
    logger.info("Added %d chunks to the vector store.", len(docs))
    return len(docs)


def reset_collection() -> None:
    """Delete every document in the collection."""
    store = get_vector_store()
    store.delete_collection()
    logger.info("Vector store collection cleared.")


def count_documents() -> int:
    """Return the number of vectors currently stored."""
    store = get_vector_store()
    try:
        return store._collection.count()  # noqa: SLF001 - Chroma exposes this only privately
    except Exception:  # noqa: BLE001
        return 0
