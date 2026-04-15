"""Shared backend factory for benchmark runners."""

from __future__ import annotations

from src.backends.ollama_backend import OllamaBackend
from src.backends.vllm_backend import VLLMBackend


def make_backend(backend_name: str, base_url: str, timeout: int) -> OllamaBackend | VLLMBackend:
    """Create a backend client by name."""
    if backend_name == "ollama":
        return OllamaBackend(base_url=base_url, timeout=timeout)
    if backend_name == "vllm":
        return VLLMBackend(base_url=base_url, timeout=timeout)
    raise ValueError(f"Unsupported backend: {backend_name}")
