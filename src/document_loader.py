"""Document loaders for PDFs, plain text, DOCX, and web URLs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}


def load_file(path: str | Path) -> list[Document]:
    """Load a single file and return LangChain `Document` objects."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
    elif suffix in {".txt", ".md"}:
        loader = TextLoader(str(path), encoding="utf-8")
    elif suffix == ".docx":
        loader = Docx2txtLoader(str(path))
    else:
        raise ValueError(
            f"Unsupported file type '{suffix}'. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    docs = loader.load()
    for doc in docs:
        doc.metadata.setdefault("source", str(path))
        doc.metadata.setdefault("source_type", "file")
        doc.metadata.setdefault("filename", path.name)
    logger.info("Loaded %d document(s) from %s", len(docs), path)
    return docs


def load_directory(directory: str | Path) -> list[Document]:
    """Recursively load every supported file from a directory."""
    directory = Path(directory)
    if not directory.exists() or not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    docs: list[Document] = []
    for file_path in directory.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            try:
                docs.extend(load_file(file_path))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping %s: %s", file_path, exc)
    return docs


def load_url(url: str, timeout: int = 30) -> list[Document]:
    """Fetch a URL and extract clean text."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; RAGBot/1.0; +https://example.com/bot)"
        )
    }
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()

    text = "\n".join(line.strip() for line in soup.get_text().splitlines() if line.strip())
    title = soup.title.string.strip() if soup.title and soup.title.string else url

    return [
        Document(
            page_content=text,
            metadata={"source": url, "source_type": "url", "title": title},
        )
    ]


def load_urls(urls: Iterable[str]) -> list[Document]:
    """Load multiple URLs, skipping any that fail."""
    documents: list[Document] = []
    for url in urls:
        try:
            documents.extend(load_url(url))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load %s: %s", url, exc)
    return documents
