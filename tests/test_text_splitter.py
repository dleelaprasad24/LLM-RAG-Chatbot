"""Tests for the text splitter."""

from __future__ import annotations

from langchain_core.documents import Document

from src.text_splitter import split_documents


def test_split_documents_creates_overlapping_chunks() -> None:
    long_text = ("This is a sentence. " * 200).strip()
    docs = [Document(page_content=long_text, metadata={"source": "test.txt"})]

    chunks = split_documents(docs, chunk_size=200, chunk_overlap=50)

    assert len(chunks) > 1
    assert all(isinstance(c, Document) for c in chunks)
    assert all(c.metadata.get("source") == "test.txt" for c in chunks)
    assert all(len(c.page_content) <= 200 for c in chunks)


def test_split_documents_empty_input() -> None:
    assert split_documents([]) == []
