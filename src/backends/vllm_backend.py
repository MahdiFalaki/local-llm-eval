"""Minimal optional vLLM backend using an OpenAI-compatible server."""

from __future__ import annotations

import importlib.util
import logging
import time

import requests

from src.backends.ollama_backend import GenerationRequest, GenerationResult
from src.utils import estimate_token_count

LOGGER = logging.getLogger(__name__)


class VLLMBackend:
    """Call a running vLLM OpenAI-compatible server."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @staticmethod
    def is_installed() -> bool:
        """Return whether the `vllm` Python package is importable."""
        return importlib.util.find_spec("vllm") is not None

    def is_reachable(self) -> bool:
        """Return whether the vLLM server appears reachable."""
        for path in ("/health", "/v1/models"):
            try:
                response = requests.get(f"{self.base_url}{path}", timeout=3)
                if response.ok:
                    return True
            except requests.RequestException:
                continue
        return False

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Run a completion request against the vLLM server."""
        payload = {
            "model": request.model_name,
            "prompt": request.prompt,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        start = time.perf_counter()
        try:
            response = requests.post(
                f"{self.base_url}/v1/completions",
                json=payload,
                timeout=request.timeout,
            )
            response.raise_for_status()
            raw = response.json()
        except requests.Timeout as exc:
            message = f"vLLM request timed out after {request.timeout}s"
            LOGGER.error(message)
            return GenerationResult("", 0.0, 0, 0.0, error=message, raw={"exception": str(exc)})
        except requests.RequestException as exc:
            message = f"vLLM request failed: {exc}"
            LOGGER.error(message)
            return GenerationResult("", 0.0, 0, 0.0, error=message, raw={"exception": str(exc)})

        latency = time.perf_counter() - start
        choices = raw.get("choices") or []
        text = str(choices[0].get("text", "")).strip() if choices else ""
        usage = raw.get("usage") or {}
        output_tokens = int(usage.get("completion_tokens") or estimate_token_count(text))
        tokens_per_second = output_tokens / latency if latency > 0 else 0.0

        return GenerationResult(
            text=text,
            latency_seconds=round(latency, 4),
            output_tokens=output_tokens,
            tokens_per_second=round(tokens_per_second, 4),
            raw=raw,
            error=None,
        )

