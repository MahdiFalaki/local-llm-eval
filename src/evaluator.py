"""Evaluation loop for model and prompt combinations."""

from __future__ import annotations

import csv
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from src.backends.ollama_backend import GenerationRequest, GenerationResult, OllamaBackend
from src.backends.vllm_backend import VLLMBackend
from src.rubric import evaluate_response
from src.utils import build_prompt_text, ensure_directory, estimate_token_count, load_yaml, shorten, utc_now_iso, write_json

LOGGER = logging.getLogger(__name__)


class BackendProtocol(Protocol):
    """Protocol for backend clients."""

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate text for a prompt."""


@dataclass(slots=True)
class ModelConfig:
    """Model definition loaded from YAML."""

    id: str
    backend: str
    model_name: str
    enabled: bool
    timeout: int


@dataclass(slots=True)
class PromptConfig:
    """Prompt definition loaded from YAML."""

    id: str
    category: str
    prompt: str
    context: str
    temperatures: list[float]
    max_tokens: int
    expected_keywords: list[str]
    required_substrings: list[str]


@dataclass(slots=True)
class EvaluationRecord:
    """One saved evaluation row."""

    run_at_utc: str
    backend: str
    model_id: str
    model_name: str
    prompt_id: str
    category: str
    temperature: float
    max_tokens: int
    latency_seconds: float
    output_tokens: int
    tokens_per_second: float
    response_length_chars: int
    response_length_words: int
    response_length_estimated_tokens: int
    rubric_score: float
    rubric_non_empty: bool
    rubric_format_compliance: bool
    rubric_keyword_hits: int
    rubric_keyword_total: int
    rubric_keyword_coverage: float
    response_preview: str
    response_text: str
    error: str | None


def load_model_configs(path: Path, backend_filter: str | None = None) -> list[ModelConfig]:
    """Load and filter models from YAML."""
    payload = load_yaml(path)
    models = payload.get("models", [])
    configs: list[ModelConfig] = []
    for item in models:
        if not item.get("enabled", True):
            continue
        config = ModelConfig(
            id=str(item["id"]),
            backend=str(item["backend"]),
            model_name=str(item["model_name"]),
            enabled=bool(item.get("enabled", True)),
            timeout=int(item.get("timeout", payload.get("defaults", {}).get("timeout", 120))),
        )
        if backend_filter and config.backend != backend_filter:
            continue
        configs.append(config)
    if not configs:
        raise ValueError(f"No enabled models found in {path} for backend={backend_filter!r}")
    return configs


def load_prompt_configs(path: Path) -> list[PromptConfig]:
    """Load prompts from YAML."""
    payload = load_yaml(path)
    prompts = payload.get("prompts", [])
    configs: list[PromptConfig] = []
    for item in prompts:
        configs.append(
            PromptConfig(
                id=str(item["id"]),
                category=str(item["category"]),
                prompt=str(item["prompt"]),
                context=str(item.get("context", "")),
                temperatures=[float(value) for value in item.get("temperatures", [0.2])],
                max_tokens=int(item.get("max_tokens", 128)),
                expected_keywords=[str(value) for value in item.get("expected_keywords", [])],
                required_substrings=[str(value) for value in item.get("required_substrings", [])],
            )
        )
    if not configs:
        raise ValueError(f"No prompts found in {path}")
    return configs


class Evaluator:
    """Coordinate evaluation runs and save structured results."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = ensure_directory(output_dir)

    def _make_backend(self, backend_name: str, base_url: str, timeout: int) -> BackendProtocol:
        if backend_name == "ollama":
            return OllamaBackend(base_url=base_url, timeout=timeout)
        if backend_name == "vllm":
            return VLLMBackend(base_url=base_url, timeout=timeout)
        raise ValueError(f"Unsupported backend: {backend_name}")

    def run(
        self,
        models: list[ModelConfig],
        prompts: list[PromptConfig],
        backend_urls: dict[str, str],
    ) -> list[EvaluationRecord]:
        """Run all selected model and prompt combinations."""
        results: list[EvaluationRecord] = []
        for model in models:
            backend = self._make_backend(model.backend, backend_urls[model.backend], model.timeout)
            for prompt in prompts:
                prompt_text = build_prompt_text(prompt.prompt, prompt.context)
                for temperature in prompt.temperatures:
                    LOGGER.info(
                        "Running model=%s prompt=%s temperature=%.2f",
                        model.id,
                        prompt.id,
                        temperature,
                    )
                    response = backend.generate(
                        GenerationRequest(
                            model_name=model.model_name,
                            prompt=prompt_text,
                            temperature=temperature,
                            max_tokens=prompt.max_tokens,
                            timeout=model.timeout,
                        )
                    )
                    rubric = evaluate_response(
                        response.text,
                        expected_keywords=prompt.expected_keywords,
                        required_substrings=prompt.required_substrings,
                    )
                    response_words = len(response.text.split())
                    record = EvaluationRecord(
                        run_at_utc=utc_now_iso(),
                        backend=model.backend,
                        model_id=model.id,
                        model_name=model.model_name,
                        prompt_id=prompt.id,
                        category=prompt.category,
                        temperature=temperature,
                        max_tokens=prompt.max_tokens,
                        latency_seconds=response.latency_seconds,
                        output_tokens=response.output_tokens,
                        tokens_per_second=response.tokens_per_second,
                        response_length_chars=len(response.text),
                        response_length_words=response_words,
                        response_length_estimated_tokens=estimate_token_count(response.text),
                        rubric_score=rubric.score,
                        rubric_non_empty=rubric.non_empty,
                        rubric_format_compliance=rubric.format_compliance,
                        rubric_keyword_hits=rubric.keyword_hits,
                        rubric_keyword_total=rubric.keyword_total,
                        rubric_keyword_coverage=round(rubric.keyword_coverage, 4),
                        response_preview=shorten(response.text),
                        response_text=response.text,
                        error=response.error,
                    )
                    results.append(record)
        return results

    def save(
        self,
        records: list[EvaluationRecord],
        environment: dict[str, Any],
        models_path: Path,
        prompts_path: Path,
    ) -> tuple[Path, Path]:
        """Save CSV and JSON outputs."""
        csv_path = self.output_dir / "results.csv"
        json_path = self.output_dir / "results.json"
        fieldnames = list(asdict(records[0]).keys()) if records else []

        if records:
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for record in records:
                    writer.writerow(asdict(record))
        else:
            csv_path.write_text("", encoding="utf-8")

        summary = {
            "generated_at_utc": utc_now_iso(),
            "models_config": str(models_path),
            "prompts_config": str(prompts_path),
            "record_count": len(records),
        }
        payload = {
            "summary": summary,
            "environment": environment,
            "results": [asdict(record) for record in records],
        }
        write_json(json_path, payload)
        return csv_path, json_path

