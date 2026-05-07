"""API smoke tests using FastAPI's in-process TestClient.

These tests exercise every endpoint that does NOT require an LLM call, plus a
mocked /chat to verify request/response wiring without spending API credits.
"""

from __future__ import annotations

from io import BytesIO
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.main import app, get_chatbot
from src.rag_chain import RAGResponse


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_redirects_to_docs(client: TestClient) -> None:
    response = client.get("/", follow_redirects=False)
    assert response.status_code in {307, 308}
    assert response.headers["location"] == "/docs"


def test_status_returns_config(client: TestClient) -> None:
    response = client.get("/status")
    assert response.status_code == 200

    body = response.json()
    assert body["provider"] in {"groq", "openai", "ollama"}
    assert body["chunk_size"] > 0
    assert body["chunk_overlap"] >= 0
    assert body["top_k"] >= 1
    assert "embedding_model" in body
    assert isinstance(body["vectors_count"], int)


def test_ingest_files_rejects_unsupported_extension(client: TestClient) -> None:
    response = client.post(
        "/ingest/files",
        files=[("files", ("evil.exe", BytesIO(b"MZ\x90\x00"), "application/octet-stream"))],
    )
    assert response.status_code == 400
    assert "Unsupported" in response.json()["detail"]


def test_ingest_urls_validates_payload(client: TestClient) -> None:
    response = client.post("/ingest/urls", json={"urls": []})
    assert response.status_code == 422


def test_chat_returns_409_when_collection_empty(client: TestClient) -> None:
    """If no docs are ingested, /chat should fail loudly with 409 - not 500."""
    with patch("api.main.count_documents", return_value=0):
        response = client.post("/chat", json={"question": "hello", "history": []})
    assert response.status_code == 409
    assert "ingest" in response.json()["detail"].lower()


def test_chat_happy_path_with_mocked_llm(client: TestClient) -> None:
    """End-to-end /chat wiring: history, response shape, sources - no real LLM call."""

    class FakeBot:
        def ask(self, question, chat_history=None):
            assert question == "What is RAG?"
            assert len(chat_history) == 2
            return RAGResponse(answer="RAG = retrieve + generate.", sources=[])

    app.dependency_overrides[get_chatbot] = lambda: FakeBot()
    try:
        with patch("api.main.count_documents", return_value=5):
            response = client.post(
                "/chat",
                json={
                    "question": "What is RAG?",
                    "history": [
                        {"role": "user", "content": "hi"},
                        {"role": "assistant", "content": "hello!"},
                    ],
                },
            )
        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == "RAG = retrieve + generate."
        assert body["sources"] == []
    finally:
        app.dependency_overrides.clear()


def test_chat_validates_empty_question(client: TestClient) -> None:
    response = client.post("/chat", json={"question": "", "history": []})
    assert response.status_code == 422
