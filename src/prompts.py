"""Prompt templates used by the RAG pipeline."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

SYSTEM_PROMPT = """You are a helpful, precise research assistant.

Answer the user's question using ONLY the information in the provided context.
Follow these rules strictly:

1. If the answer is not in the context, reply with:
   "I don't have enough information in the provided documents to answer that."
2. Be concise and factual. Prefer bullet points for lists.
3. Cite the supporting passages by referencing the source filename or URL in
   square brackets, for example: [report.pdf] or [https://example.com].
4. Never invent facts, names, dates, numbers, or citations.

----- CONTEXT -----
{context}
-------------------
"""

CONDENSE_QUESTION_PROMPT = """Given the chat history below and a follow-up
question, rewrite the follow-up so that it is a standalone question that can
be understood without the chat history. Preserve the user's original intent
and language. Return ONLY the rewritten question - no preamble.

Chat history:
{chat_history}

Follow-up question:
{question}

Standalone question:"""


def build_qa_prompt() -> ChatPromptTemplate:
    """Return the chat prompt used to answer questions with retrieved context."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{question}"),
        ]
    )


def build_condense_prompt() -> ChatPromptTemplate:
    """Return the prompt that turns a follow-up into a standalone question."""
    return ChatPromptTemplate.from_template(CONDENSE_QUESTION_PROMPT)
