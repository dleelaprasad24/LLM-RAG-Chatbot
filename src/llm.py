"""LLM provider factory.

Supports Groq (default - fast & free), OpenAI, and Ollama (fully local).
The active provider is selected via the ``LLM_PROVIDER`` environment variable.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from .config import get_settings


def get_llm() -> BaseChatModel:
    """Build a chat model for the configured provider."""
    settings = get_settings()
    provider = settings.llm_provider.lower()

    if provider == "groq":
        if not settings.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is missing. Add it to your .env file or "
                "switch LLM_PROVIDER to 'openai' or 'ollama'."
            )
        from langchain_groq import ChatGroq

        return ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )

    if provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is missing in your .env file.")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=settings.temperature,
            num_predict=settings.max_tokens,
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER '{provider}'. Use one of: groq, openai, ollama."
    )
