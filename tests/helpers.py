"""Shared test helpers for local-llm-eval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def create_prompt_run(run_dir: Path) -> Path:
    """Create a minimal prompt-suite run fixture."""
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": {
            "generated_at_utc": "2026-04-15T00:00:00+00:00",
            "record_count": 2,
        },
        "environment": {"runtime": {"python_version": "3.11.0"}},
        "results": [
            {
                "model_id": "qwen2_5_7b",
                "model_name": "qwen2.5:7b",
                "backend": "ollama",
                "prompt_id": "p1",
                "category": "format",
                "temperature": 0.2,
                "latency_seconds": 1.25,
                "tokens_per_second": 50.0,
                "response_length_words": 12,
                "rubric_score": 0.75,
                "rubric_format_compliance": True,
                "rubric_keyword_total": 2,
                "rubric_keyword_hits": 2,
                "response_text": "answer one",
                "error": None,
            },
            {
                "model_id": "qwen2_5_7b",
                "model_name": "qwen2.5:7b",
                "backend": "ollama",
                "prompt_id": "p2",
                "category": "keywords",
                "temperature": 0.2,
                "latency_seconds": 2.0,
                "tokens_per_second": 40.0,
                "response_length_words": 8,
                "rubric_score": 0.25,
                "rubric_format_compliance": False,
                "rubric_keyword_total": 2,
                "rubric_keyword_hits": 1,
                "response_text": "short",
                "error": "backend failed",
            },
        ],
    }
    (run_dir / "results.json").write_text(json.dumps(payload), encoding="utf-8")
    (run_dir / "config_snapshot.json").write_text(json.dumps({"benchmark": "prompts"}), encoding="utf-8")
    return run_dir / "results.json"


def create_ifeval_run(run_dir: Path) -> Path:
    """Create a minimal IFEval run fixture."""
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "generated_at_utc": "2026-04-15T00:00:00+00:00",
        "benchmark": "ifeval",
        "record_count": 3,
        "models": {
            "qwen2_5_7b": {
                "model_name": "qwen2.5:7b",
                "backend": "ollama",
                "prompt_count": 3,
                "instruction_count": 4,
                "prompt_strict_accuracy": 0.666667,
                "prompt_loose_accuracy": 1.0,
                "instruction_strict_accuracy": 0.75,
                "instruction_loose_accuracy": 1.0,
                "average_latency_seconds": 1.2,
                "average_tokens_per_second": 80.0,
            }
        },
    }
    records: list[dict[str, Any]] = [
        {
            "model_id": "qwen2_5_7b",
            "model_name": "qwen2.5:7b",
            "backend": "ollama",
            "example_key": 1,
            "prompt": "Write JSON only",
            "instruction_id_list": ["detectable_format:json"],
            "instruction_count": 1,
            "temperature": 0.0,
            "max_tokens": 128,
            "latency_seconds": 1.0,
            "output_tokens": 20,
            "tokens_per_second": 100.0,
            "prompt_strict": True,
            "prompt_loose": True,
            "instruction_strict_followed": 1,
            "instruction_loose_followed": 1,
            "instruction_strict_list": [True],
            "instruction_loose_list": [True],
            "response_text": "{\"ok\": true}",
            "error": None,
        },
        {
            "model_id": "qwen2_5_7b",
            "model_name": "qwen2.5:7b",
            "backend": "ollama",
            "example_key": 2,
            "prompt": "Use the keyword alpha",
            "instruction_id_list": ["keywords:include"],
            "instruction_count": 1,
            "temperature": 0.0,
            "max_tokens": 128,
            "latency_seconds": 1.5,
            "output_tokens": 5,
            "tokens_per_second": 50.0,
            "prompt_strict": False,
            "prompt_loose": True,
            "instruction_strict_followed": 0,
            "instruction_loose_followed": 1,
            "instruction_strict_list": [False],
            "instruction_loose_list": [True],
            "response_text": "tiny",
            "error": None,
        },
        {
            "model_id": "qwen2_5_7b",
            "model_name": "qwen2.5:7b",
            "backend": "ollama",
            "example_key": 3,
            "prompt": "Use English and no commas",
            "instruction_id_list": ["language:en", "punctuation:no_comma"],
            "instruction_count": 2,
            "temperature": 0.0,
            "max_tokens": 128,
            "latency_seconds": 1.1,
            "output_tokens": 30,
            "tokens_per_second": 90.0,
            "prompt_strict": True,
            "prompt_loose": True,
            "instruction_strict_followed": 2,
            "instruction_loose_followed": 2,
            "instruction_strict_list": [True, True],
            "instruction_loose_list": [True, True],
            "response_text": "A correct answer without commas",
            "error": None,
        },
    ]
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run_dir / "config_snapshot.json").write_text(json.dumps({"benchmark": "ifeval"}), encoding="utf-8")
    with (run_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record))
            handle.write("\n")
    with (run_dir / "responses.jsonl").open("w", encoding="utf-8") as handle:
        handle.write("")
    return run_dir
