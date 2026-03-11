"""Shared OpenAI client setup used by phases 2.5 and 4."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import openai
except ImportError:
    raise SystemExit(
        "Missing dependency: openai. Install with: pip install openai"
    )


def load_api_key(explicit_path: Optional[str] = None) -> str:
    """Resolve an OpenAI API key.

    Priority:
    1. ``OPENAI_API_KEY`` environment variable (supports ``.env`` via python-dotenv)
    2. Explicit key-file path (if provided)
    3. ``OPENAIKEY.txt`` / ``openaikey.txt`` in cwd or script directory
    """
    # 1. Environment variable
    env_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key

    # 2 & 3. File-based lookup
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser().resolve())

    cwd = Path.cwd()
    candidates.extend([
        cwd / "OPENAIKEY.txt",
        cwd / "openaikey.txt",
        Path(__file__).resolve().parent / "OPENAIKEY.txt",
    ])

    for path in candidates:
        if path.exists() and path.is_file():
            key = path.read_text(encoding="utf-8").strip()
            if key:
                return key

    raise FileNotFoundError(
        "No OpenAI API key found. Set OPENAI_API_KEY in .env, "
        "or place OPENAIKEY.txt in the working directory."
    )


def get_openai_client(api_key: Optional[str] = None) -> openai.OpenAI:
    """Return a configured ``openai.OpenAI`` client."""
    key = api_key or load_api_key()
    return openai.OpenAI(api_key=key)
