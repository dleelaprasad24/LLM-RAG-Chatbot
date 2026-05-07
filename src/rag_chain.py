"""Conversational RAG pipeline.

The pipeline:
    1. Optionally rewrites a follow-up question into a standalone query
       using the chat history (history-aware retrieval).
    2. Retrieves the top-K most relevant chunks from the Chroma vector store.
    3. Feeds the retrieved context + chat history + question into the LLM
       using a strict, citation-aware system prompt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from .config import get_settings
from .llm import get_llm
from .prompts import build_condense_prompt, build_qa_prompt
from .vector_store import get_vector_store


@dataclass
class RAGResponse:
    """Container for an answer plus the documents that supported it."""

    answer: str
    sources: list[Document] = field(default_factory=list)

    def formatted_sources(self) -> list[dict[str, Any]]:
        """Return source metadata in a UI-friendly shape."""
        formatted: list[dict[str, Any]] = []
        for doc in self.sources:
            meta = doc.metadata or {}
            formatted.append(
                {
                    "source": meta.get("source", "unknown"),
                    "page": meta.get("page"),
                    "title": meta.get("title") or meta.get("filename"),
                    "snippet": doc.page_content[:300].strip(),
                }
            )
        return formatted


def _format_docs(docs: list[Document]) -> str:
    """Render retrieved documents into a single context string for the LLM."""
    if not docs:
        return "No documents retrieved."
    blocks = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        source = meta.get("source", "unknown")
        page = f", page {meta['page'] + 1}" if isinstance(meta.get("page"), int) else ""
        blocks.append(
            f"[Doc {i} | source: {source}{page}]\n{doc.page_content.strip()}"
        )
    return "\n\n".join(blocks)


def _format_history(history: list[BaseMessage]) -> str:
    """Turn a list of chat messages into a plain-text transcript."""
    lines: list[str] = []
    for msg in history:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines) if lines else "(no prior conversation)"


class RAGChatbot:
    """High-level interface for the conversational RAG pipeline."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.llm = get_llm()
        self.vector_store = get_vector_store()
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.settings.top_k},
        )
        self._qa_prompt = build_qa_prompt()
        self._condense_prompt = build_condense_prompt()
        self._condense_chain = (
            self._condense_prompt | self.llm | StrOutputParser()
        )

    def _rewrite_question(self, question: str, history: list[BaseMessage]) -> str:
        """Rewrite a follow-up so it is standalone (history-aware retrieval)."""
        if not history:
            return question
        return self._condense_chain.invoke(
            {"question": question, "chat_history": _format_history(history)}
        ).strip()

    def ask(
        self,
        question: str,
        chat_history: list[BaseMessage] | None = None,
    ) -> RAGResponse:
        """Answer a question, optionally using prior conversation turns."""
        history = chat_history or []
        standalone = self._rewrite_question(question, history)

        retrieved_docs = self.retriever.invoke(standalone)

        chain = (
            RunnablePassthrough.assign(
                context=RunnableLambda(lambda _: _format_docs(retrieved_docs))
            )
            | self._qa_prompt
            | self.llm
            | StrOutputParser()
        )

        answer = chain.invoke({"question": question, "chat_history": history})
        return RAGResponse(answer=answer.strip(), sources=retrieved_docs)


def build_chat_history(turns: list[tuple[str, str]]) -> list[BaseMessage]:
    """Convert ``[(user, assistant), ...]`` tuples into LangChain messages."""
    history: list[BaseMessage] = []
    for user_msg, assistant_msg in turns:
        history.append(HumanMessage(content=user_msg))
        history.append(AIMessage(content=assistant_msg))
    return history
