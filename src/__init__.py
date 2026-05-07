"""LLM-Powered RAG Chatbot package.

This module performs an early sqlite3 shim: ChromaDB requires sqlite3 >= 3.35
which many older Linux distros (e.g. Ubuntu 20.04) ship without. If the
``pysqlite3-binary`` wheel is installed, swap it in before chromadb gets a
chance to import the system ``sqlite3`` module. This is the workaround
documented by ChromaDB itself.
"""

from __future__ import annotations

import sys

try:
    import pysqlite3  # type: ignore[import-not-found]

    sys.modules["sqlite3"] = pysqlite3
except ModuleNotFoundError:
    pass
