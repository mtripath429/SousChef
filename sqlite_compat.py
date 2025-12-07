"""
SQLite compatibility shim for environments where the system sqlite3 is too old
for ChromaDB (requires >= 3.35.0). On platforms like Streamlit Cloud, we can
vendor a modern SQLite via the pysqlite3-binary package and alias it as
the standard library's sqlite3 module before Chroma is imported.

This module should be imported as early as possible (before importing chromadb).
"""

import os
import sys


def _ensure_modern_sqlite() -> None:
    # Allow disabling via env if needed for debugging
    if os.getenv("DISABLE_PYSQLITE3_SHIM"):
        return

    try:
        # If sqlite3 already imported, do nothing
        if "sqlite3" in sys.modules:
            return
        # Try to import pysqlite3 and alias it
        __import__("pysqlite3")
        sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
    except Exception:
        # Silently continue; chromadb will raise a helpful error if insufficient
        # version remains. This keeps local dev flexible if system sqlite is ok.
        pass


_ensure_modern_sqlite()
