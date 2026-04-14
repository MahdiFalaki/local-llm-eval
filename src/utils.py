"""Shared helpers for configuration, paths, and lightweight metrics."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file into a Python dictionary."""
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}, got {type(data).__name__}")
    return data


def write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    """Write JSON with stable indentation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def ensure_directory(path: Path) -> Path:
    """Create a directory if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def estimate_token_count(text: str) -> int:
    """Return a rough token estimate from natural language text."""
    words = re.findall(r"\S+", text)
    if not words:
        return 0
    return max(1, round(len(words) * 1.3))


def utc_now_iso() -> str:
    """Return a UTC timestamp string."""
    return datetime.now(timezone.utc).isoformat()


def build_prompt_text(prompt: str, context: str | None = None) -> str:
    """Compose the final prompt text sent to a backend."""
    if not context:
        return prompt.strip()
    return f"{prompt.strip()}\n\nContext:\n{context.strip()}"


def shorten(text: str, limit: int = 160) -> str:
    """Create a compact preview string for logs and outputs."""
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 3]}..."

