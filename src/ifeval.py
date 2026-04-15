"""IFEval dataset loading, evaluation, and output writing."""

from __future__ import annotations

import csv
import importlib
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.backends.factory import make_backend
from src.backends.ollama_backend import GenerationRequest
from src.evaluator import ModelConfig
from src.utils import ensure_directory, sanitize_name, shorten, utc_now_iso, write_json, write_jsonl

LOGGER = logging.getLogger(__name__)


def _load_dataset_function() -> Any:
    """Return the Hugging Face dataset loader when available."""
    try:
        datasets_module = importlib.import_module("datasets")
    except ImportError as exc:
        raise RuntimeError(
            "IFEval support requires the optional `datasets` package. "
            "Install the project dependencies or use a non-IFEval workflow."
        ) from exc
    return datasets_module.load_dataset


def _load_evaluation_lib() -> Any:
    """Import the vendored IFEval checker only when benchmark execution needs it."""
    try:
        return importlib.import_module("src.ifeval_official.evaluation_lib")
    except ImportError as exc:
        raise RuntimeError(
            "IFEval support requires its benchmark dependencies to be installed "
            "(for example `nltk`, `langdetect`, and `immutabledict`)."
        ) from exc


@dataclass(slots=True)
class IFEvalExample:
    """One dataset example from google/IFEval."""

    key: int
    prompt: str
    instruction_id_list: list[str]
    kwargs: list[dict[str, Any]]


@dataclass(slots=True)
class IFEvalResponseRecord:
    """Per-example raw generation output."""

    run_at_utc: str
    model_id: str
    model_name: str
    backend: str
    example_key: int
    prompt: str
    instruction_id_list: list[str]
    kwargs: list[dict[str, Any]]
    temperature: float
    max_tokens: int
    latency_seconds: float
    output_tokens: int
    tokens_per_second: float
    response_preview: str
    response_text: str
    error: str | None


@dataclass(slots=True)
class IFEvalResultRecord:
    """Per-example benchmark result with strict and loose scores."""

    run_at_utc: str
    model_id: str
    model_name: str
    backend: str
    example_key: int
    prompt: str
    instruction_id_list: list[str]
    instruction_count: int
    temperature: float
    max_tokens: int
    latency_seconds: float
    output_tokens: int
    tokens_per_second: float
    prompt_strict: bool
    prompt_loose: bool
    instruction_strict_followed: int
    instruction_loose_followed: int
    instruction_strict_list: list[bool]
    instruction_loose_list: list[bool]
    response_preview: str
    response_text: str
    error: str | None


def load_ifeval_examples(limit: int | None = None) -> list[IFEvalExample]:
    """Load the official google/IFEval dataset."""
    load_dataset = _load_dataset_function()
    dataset = load_dataset("google/IFEval", split="train")
    examples: list[IFEvalExample] = []
    for index, row in enumerate(dataset):
        if limit is not None and index >= limit:
            break
        examples.append(
            IFEvalExample(
                key=int(row["key"]),
                prompt=str(row["prompt"]),
                instruction_id_list=[str(value) for value in row["instruction_id_list"]],
                kwargs=[
                    {key: value for key, value in dict(item).items() if value is not None}
                    for item in row["kwargs"]
                ],
            )
        )
    return examples


def dataset_summary(examples: list[IFEvalExample]) -> dict[str, Any]:
    """Build a compact dataset summary."""
    instruction_type_totals: dict[str, int] = defaultdict(int)
    instructions_per_prompt: dict[int, int] = defaultdict(int)
    for example in examples:
        instructions_per_prompt[len(example.instruction_id_list)] += 1
        for instruction_id in example.instruction_id_list:
            instruction_type_totals[instruction_id.split(":")[0]] += 1
    return {
        "dataset_name": "google/IFEval",
        "split": "train",
        "example_count": len(examples),
        "instructions_per_prompt": dict(sorted(instructions_per_prompt.items())),
        "instruction_type_totals": dict(sorted(instruction_type_totals.items())),
    }


class IFEvalRunner:
    """Run IFEval against local models and save run artifacts."""

    def __init__(self, output_root: Path) -> None:
        self.output_root = ensure_directory(output_root)

    def run(
        self,
        models: list[ModelConfig],
        examples: list[IFEvalExample],
        backend_urls: dict[str, str],
        temperature: float,
        max_tokens: int,
    ) -> tuple[list[IFEvalResponseRecord], list[IFEvalResultRecord]]:
        """Generate responses and score them with official-style strict and loose checks."""
        evaluation_lib = _load_evaluation_lib()
        response_records: list[IFEvalResponseRecord] = []
        result_records: list[IFEvalResultRecord] = []

        for model in models:
            backend = make_backend(model.backend, backend_urls[model.backend], model.timeout)
            for example in examples:
                LOGGER.info("IFEval model=%s example=%s", model.id, example.key)
                generation = backend.generate(
                    GenerationRequest(
                        model_name=model.model_name,
                        prompt=example.prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        timeout=model.timeout,
                    )
                )
                response_record = IFEvalResponseRecord(
                    run_at_utc=utc_now_iso(),
                    model_id=model.id,
                    model_name=model.model_name,
                    backend=model.backend,
                    example_key=example.key,
                    prompt=example.prompt,
                    instruction_id_list=example.instruction_id_list,
                    kwargs=example.kwargs,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    latency_seconds=generation.latency_seconds,
                    output_tokens=generation.output_tokens,
                    tokens_per_second=generation.tokens_per_second,
                    response_preview=shorten(generation.text),
                    response_text=generation.text,
                    error=generation.error,
                )
                response_records.append(response_record)

                official_input = evaluation_lib.InputExample(
                    key=example.key,
                    instruction_id_list=example.instruction_id_list,
                    prompt=example.prompt,
                    kwargs=example.kwargs,
                )
                prompt_to_response = {example.prompt: generation.text}
                strict_output = evaluation_lib.test_instruction_following_strict(official_input, prompt_to_response)
                loose_output = evaluation_lib.test_instruction_following_loose(official_input, prompt_to_response)

                result_records.append(
                    IFEvalResultRecord(
                        run_at_utc=response_record.run_at_utc,
                        model_id=model.id,
                        model_name=model.model_name,
                        backend=model.backend,
                        example_key=example.key,
                        prompt=example.prompt,
                        instruction_id_list=example.instruction_id_list,
                        instruction_count=len(example.instruction_id_list),
                        temperature=temperature,
                        max_tokens=max_tokens,
                        latency_seconds=generation.latency_seconds,
                        output_tokens=generation.output_tokens,
                        tokens_per_second=generation.tokens_per_second,
                        prompt_strict=bool(strict_output.follow_all_instructions),
                        prompt_loose=bool(loose_output.follow_all_instructions),
                        instruction_strict_followed=sum(strict_output.follow_instruction_list),
                        instruction_loose_followed=sum(loose_output.follow_instruction_list),
                        instruction_strict_list=list(strict_output.follow_instruction_list),
                        instruction_loose_list=list(loose_output.follow_instruction_list),
                        response_preview=response_record.response_preview,
                        response_text=generation.text,
                        error=generation.error,
                    )
                )
        return response_records, result_records

    def save(
        self,
        run_name: str,
        response_records: list[IFEvalResponseRecord],
        result_records: list[IFEvalResultRecord],
        config_snapshot: dict[str, Any],
        environment: dict[str, Any],
        dataset_info: dict[str, Any],
    ) -> Path:
        """Save the benchmark run in a dedicated run directory."""
        run_dir = ensure_directory(self.output_root / sanitize_name(run_name))
        responses_path = run_dir / "responses.jsonl"
        results_path = run_dir / "results.jsonl"
        summary_path = run_dir / "summary.json"
        config_snapshot_path = run_dir / "config_snapshot.json"
        csv_path = run_dir / "results.csv"

        write_jsonl(responses_path, (asdict(record) for record in response_records))
        write_jsonl(results_path, (asdict(record) for record in result_records))
        self._write_csv(csv_path, result_records)

        summary = self._build_summary(result_records)
        summary["dataset"] = dataset_info
        summary["environment"] = environment
        write_json(summary_path, summary)
        write_json(config_snapshot_path, config_snapshot)
        return run_dir

    def _write_csv(self, path: Path, records: list[IFEvalResultRecord]) -> None:
        if not records:
            path.write_text("", encoding="utf-8")
            return
        rows = [asdict(record) for record in records]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def _build_summary(self, records: list[IFEvalResultRecord]) -> dict[str, Any]:
        model_summary: dict[str, dict[str, Any]] = {}
        grouped: dict[str, list[IFEvalResultRecord]] = defaultdict(list)
        for record in records:
            grouped[record.model_id].append(record)

        for model_id, model_records in grouped.items():
            prompt_total = len(model_records)
            instruction_total = sum(record.instruction_count for record in model_records)
            prompt_strict_correct = sum(int(record.prompt_strict) for record in model_records)
            prompt_loose_correct = sum(int(record.prompt_loose) for record in model_records)
            instruction_strict_correct = sum(record.instruction_strict_followed for record in model_records)
            instruction_loose_correct = sum(record.instruction_loose_followed for record in model_records)

            per_instruction: dict[str, dict[str, int]] = defaultdict(lambda: {
                "total": 0,
                "strict_correct": 0,
                "loose_correct": 0,
            })
            for record in model_records:
                for instruction_id, strict_ok, loose_ok in zip(
                    record.instruction_id_list,
                    record.instruction_strict_list,
                    record.instruction_loose_list,
                ):
                    bucket = per_instruction[instruction_id]
                    bucket["total"] += 1
                    bucket["strict_correct"] += int(strict_ok)
                    bucket["loose_correct"] += int(loose_ok)

            model_summary[model_id] = {
                "model_name": model_records[0].model_name,
                "backend": model_records[0].backend,
                "prompt_count": prompt_total,
                "instruction_count": instruction_total,
                "prompt_strict_accuracy": round(prompt_strict_correct / prompt_total, 6),
                "prompt_loose_accuracy": round(prompt_loose_correct / prompt_total, 6),
                "instruction_strict_accuracy": round(instruction_strict_correct / instruction_total, 6),
                "instruction_loose_accuracy": round(instruction_loose_correct / instruction_total, 6),
                "average_latency_seconds": round(
                    sum(record.latency_seconds for record in model_records) / prompt_total, 6
                ),
                "average_tokens_per_second": round(
                    sum(record.tokens_per_second for record in model_records) / prompt_total, 6
                ),
                "per_instruction_id": {
                    instruction_id: {
                        "total": values["total"],
                        "strict_accuracy": round(values["strict_correct"] / values["total"], 6),
                        "loose_accuracy": round(values["loose_correct"] / values["total"], 6),
                    }
                    for instruction_id, values in sorted(per_instruction.items())
                },
            }

        return {
            "generated_at_utc": utc_now_iso(),
            "benchmark": "ifeval",
            "record_count": len(records),
            "models": model_summary,
        }
