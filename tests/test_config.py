"""Tests for the application configuration."""

from __future__ import annotations

import importlib
from pathlib import Path


def test_settings_defaults(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)

    import src.config as config_module

    importlib.reload(config_module)
    settings = config_module.Settings()

    assert settings.llm_provider in {"groq", "openai", "ollama"}
    assert settings.chunk_size > 0
    assert settings.chunk_overlap >= 0
    assert settings.top_k >= 1
    assert 0.0 <= settings.temperature <= 2.0


def test_chroma_path_is_created(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "store"))

    import src.config as config_module

    importlib.reload(config_module)
    settings = config_module.Settings()

    assert settings.chroma_path.exists()
    assert settings.chroma_path.is_dir()
