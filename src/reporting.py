"""Helpers for loading completed runs and writing prompt-run snapshots."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from src.utils import ensure_directory, sanitize_name, utc_now_iso, write_json, write_jsonl


@dataclass(slots=True)
class LoadedRun:
    """Normalized representation of a completed evaluation run."""

    path: Path
    run_id: str
    benchmark: str
    label: str
    generated_at_utc: str
    records: list[dict[str, Any]]
    summary: dict[str, Any]
    config_snapshot: dict[str, Any]
    environment: dict[str, Any] | None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in {path}: {exc}") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Malformed JSONL in {path}: {exc}") from exc
    return rows


def resolve_run_path(path: Path) -> Path:
    """Resolve a user-supplied run path into a concrete artifact path."""
    candidate = path.expanduser().resolve()
    if candidate.is_dir():
        if (candidate / "summary.json").exists() and (candidate / "results.jsonl").exists():
            return candidate
        if (candidate / "results.json").exists():
            return candidate / "results.json"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Run path does not exist or is not a known artifact: {path}")


def load_run(path: Path) -> LoadedRun:
    """Load a prompt-suite or IFEval run from saved artifacts."""
    resolved = resolve_run_path(path)
    if resolved.is_file():
        if resolved.name != "results.json":
            raise ValueError(f"Unsupported run file: {resolved}")
        return _load_prompt_run(resolved)
    if (resolved / "results.jsonl").exists() and (resolved / "summary.json").exists():
        return _load_ifeval_run(resolved)
    raise ValueError(f"Unsupported run directory: {resolved}")


def _load_prompt_run(results_json_path: Path) -> LoadedRun:
    payload = _read_json(results_json_path)
    if "results" not in payload:
        raise ValueError(f"Prompt run missing `results`: {results_json_path}")
    parent = results_json_path.parent
    config_snapshot = _read_json(parent / "config_snapshot.json") if (parent / "config_snapshot.json").exists() else {}
    summary = payload.get("summary", {})
    generated_at = str(summary.get("generated_at_utc") or utc_now_iso())
    label = parent.name if parent.name != "outputs" else "prompts-legacy"
    return LoadedRun(
        path=parent,
        run_id=sanitize_name(f"{label}-{generated_at}"),
        benchmark="prompts",
        label=label,
        generated_at_utc=generated_at,
        records=[dict(record) for record in payload["results"]],
        summary=summary,
        config_snapshot=config_snapshot,
        environment=payload.get("environment"),
    )


def _load_ifeval_run(run_dir: Path) -> LoadedRun:
    summary = _read_json(run_dir / "summary.json")
    config_snapshot = _read_json(run_dir / "config_snapshot.json") if (run_dir / "config_snapshot.json").exists() else {}
    generated_at = str(summary.get("generated_at_utc") or utc_now_iso())
    return LoadedRun(
        path=run_dir,
        run_id=sanitize_name(f"{run_dir.name}-{generated_at}"),
        benchmark="ifeval",
        label=run_dir.name,
        generated_at_utc=generated_at,
        records=_read_jsonl(run_dir / "results.jsonl"),
        summary=summary,
        config_snapshot=config_snapshot,
        environment=summary.get("environment"),
    )


def default_compare_output_dir(output_root: Path) -> Path:
    """Return a default output directory for comparison artifacts."""
    return output_root / "comparisons" / sanitize_name(f"compare-{utc_now_iso().replace(':', '-')}")


def default_analysis_output_dir(output_root: Path, run: LoadedRun) -> Path:
    """Return a default output directory for analysis artifacts."""
    if run.path.is_dir():
        return run.path / "analysis"
    return output_root / "analysis" / run.run_id


def save_prompt_run_snapshot(
    output_root: Path,
    run_name: str,
    records: Iterable[Any],
    environment: dict[str, Any],
    config_snapshot: dict[str, Any],
) -> Path:
    """Write a named prompt-suite run snapshot for later comparison."""
    run_dir = ensure_directory(output_root / "prompts" / sanitize_name(run_name))
    rows = [asdict(record) if hasattr(record, "__dataclass_fields__") else dict(record) for record in records]
    csv_path = run_dir / "results.csv"
    json_path = run_dir / "results.json"
    config_path = run_dir / "config_snapshot.json"

    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("", encoding="utf-8")

    payload = {
        "summary": {
            "generated_at_utc": utc_now_iso(),
            "record_count": len(rows),
            "run_name": sanitize_name(run_name),
        },
        "environment": environment,
        "results": rows,
    }
    write_json(json_path, payload)
    write_json(config_path, config_snapshot)
    return run_dir


def write_markdown(path: Path, text: str) -> None:
    """Write a Markdown file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write CSV rows or an empty file if there are no rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
