"""Tests for document loaders."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.document_loader import load_directory, load_file


def test_load_txt_file(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.txt"
    file_path.write_text("Hello world.\nThis is a test document.", encoding="utf-8")

    docs = load_file(file_path)

    assert len(docs) == 1
    assert "Hello world" in docs[0].page_content
    assert docs[0].metadata["source"] == str(file_path)
    assert docs[0].metadata["filename"] == "sample.txt"
    assert docs[0].metadata["source_type"] == "file"


def test_load_unsupported_extension_raises(tmp_path: Path) -> None:
    file_path = tmp_path / "image.png"
    file_path.write_bytes(b"\x89PNG\r\n\x1a\n")

    with pytest.raises(ValueError):
        load_file(file_path)


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_file(tmp_path / "missing.txt")


def test_load_directory_skips_unsupported(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "b.md").write_text("# beta", encoding="utf-8")
    (tmp_path / "c.png").write_bytes(b"\x89PNG")

    docs = load_directory(tmp_path)

    sources = {Path(d.metadata["source"]).name for d in docs}
    assert sources == {"a.txt", "b.md"}
