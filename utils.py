"""Shared utility functions used across all pipeline phases."""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def safe_json_dump(obj: Any, path: Path) -> None:
    """Write *obj* as pretty-printed JSON to *path*."""
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> Any:
    """Read and parse a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(rows: list[dict], path: Path) -> None:
    """Write a list of dicts to a CSV file, preserving key insertion order."""
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def slugify(text: str, max_len: int = 120, fallback: str = "item") -> str:
    """Convert *text* to a filesystem-safe slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = re.sub(r"[-\s]+", "_", text)
    text = text.strip("_")
    return text[:max_len] or fallback


def strip_frontmatter(md: str) -> tuple[dict[str, str], str]:
    """Split YAML front-matter from a Markdown string."""
    if md.startswith("---\n"):
        parts = md.split("\n---\n", 1)
        if len(parts) == 2:
            raw_meta = parts[0][4:]
            body = parts[1]
            meta: dict[str, str] = {}
            for line in raw_meta.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip().strip('"')
            return meta, body
    return {}, md


def make_dirs(output_dir: Path, subdirs: dict[str, str]) -> dict[str, Path]:
    """Create *output_dir* and a set of named subdirectories.

    Returns a dict mapping logical names to ``Path`` objects.
    The key ``"root"`` always maps to *output_dir* itself.
    """
    dirs: dict[str, Path] = {"root": output_dir}
    for key, subpath in subdirs.items():
        dirs[key] = output_dir / subpath
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs
