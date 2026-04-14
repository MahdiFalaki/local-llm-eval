"""Minimal Ollama HTTP client backend."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import requests

from src.utils import estimate_token_count

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class GenerationRequest:
    """Single generation request shared by backends."""

    model_name: str
    prompt: str
    temperature: float
    max_tokens: int
    timeout: int


@dataclass(slots=True)
class GenerationResult:
    """Normalized generation result."""

    text: str
    latency_seconds: float
    output_tokens: int
    tokens_per_second: float
    raw: dict | None = None
    error: str | None = None


class OllamaBackend:
    """Call the local Ollama HTTP API with strict timeout handling."""

    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def is_reachable(self) -> bool:
        """Return whether the local Ollama server answers the tags endpoint."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=3)
            response.raise_for_status()
        except requests.RequestException:
            return False
        return True

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Run a non-streaming generation against Ollama."""
        payload = {
            "model": request.model_name,
            "prompt": request.prompt,
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
            },
        }
        start = time.perf_counter()
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=request.timeout,
            )
            response.raise_for_status()
            raw = response.json()
        except requests.Timeout as exc:
            message = f"Ollama request timed out after {request.timeout}s"
            LOGGER.error(message)
            return GenerationResult("", 0.0, 0, 0.0, error=message, raw={"exception": str(exc)})
        except requests.RequestException as exc:
            message = f"Ollama request failed: {exc}"
            LOGGER.error(message)
            return GenerationResult("", 0.0, 0, 0.0, error=message, raw={"exception": str(exc)})

        latency = time.perf_counter() - start
        text = str(raw.get("response", "")).strip()
        output_tokens = int(raw.get("eval_count") or estimate_token_count(text))
        eval_duration_ns = raw.get("eval_duration")
        if eval_duration_ns:
            tokens_per_second = output_tokens / (float(eval_duration_ns) / 1_000_000_000.0)
        else:
            tokens_per_second = output_tokens / latency if latency > 0 else 0.0

        return GenerationResult(
            text=text,
            latency_seconds=round(latency, 4),
            output_tokens=output_tokens,
            tokens_per_second=round(tokens_per_second, 4),
            raw=raw,
            error=None,
        )

