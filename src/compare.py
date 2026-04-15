"""Comparison and regression reporting for completed evaluation runs."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean
from typing import Any

from src.reporting import LoadedRun, load_run, write_csv, write_markdown
from src.utils import ensure_directory, write_json


@dataclass(slots=True)
class AggregateRow:
    """Aggregated metrics for one comparable slice of a run."""

    run_id: str
    run_label: str
    generated_at_utc: str
    benchmark: str
    group_type: str
    group_name: str
    model_id: str
    model_name: str
    backend: str
    record_count: int
    avg_latency_seconds: float | None
    avg_tokens_per_second: float | None
    avg_response_length_words: float | None
    avg_rubric_score: float | None
    prompt_strict_rate: float | None
    prompt_loose_rate: float | None
    instruction_strict_rate: float | None
    instruction_loose_rate: float | None
    error_rate: float | None


def compare_runs(run_paths: list[Path], output_dir: Path) -> Path:
    """Compare two or more runs and write summary artifacts."""
    if len(run_paths) < 2:
        raise ValueError("Comparison requires at least two run paths.")

    runs = [load_run(path) for path in run_paths]
    baseline = runs[0]
    baseline_rows = _aggregate_run(baseline)
    baseline_index = {_row_key(row): row for row in baseline_rows}

    comparison_rows: list[dict[str, Any]] = []
    compared_summaries: list[dict[str, Any]] = []

    for candidate in runs[1:]:
        candidate_rows = _aggregate_run(candidate)
        candidate_index = {_row_key(row): row for row in candidate_rows}
        shared_keys = sorted(set(baseline_index).intersection(candidate_index))
        rows_for_candidate: list[dict[str, Any]] = []
        metric_outcomes = {"improved": 0, "regressed": 0, "unchanged": 0}

        for key in shared_keys:
            row = _compare_aggregate_rows(baseline_index[key], candidate_index[key])
            rows_for_candidate.append(row)
            metric_outcomes[row["overall_label"]] += 1

        compared_summaries.append(
            {
                "candidate_run_id": candidate.run_id,
                "candidate_label": candidate.label,
                "shared_slice_count": len(rows_for_candidate),
                "overall_label_counts": metric_outcomes,
            }
        )
        comparison_rows.extend(rows_for_candidate)

    output_dir = ensure_directory(output_dir)
    write_json(
        output_dir / "comparison_summary.json",
        {
            "baseline": {
                "run_id": baseline.run_id,
                "label": baseline.label,
                "benchmark": baseline.benchmark,
                "generated_at_utc": baseline.generated_at_utc,
                "path": str(baseline.path),
            },
            "compared_runs": compared_summaries,
        },
    )
    write_csv(output_dir / "comparison_table.csv", comparison_rows)
    write_markdown(output_dir / "comparison_report.md", _build_report(baseline, runs[1:], comparison_rows))
    return output_dir


def _aggregate_run(run: LoadedRun) -> list[AggregateRow]:
    if run.benchmark == "prompts":
        return _aggregate_prompt_run(run)
    if run.benchmark == "ifeval":
        return _aggregate_ifeval_run(run)
    raise ValueError(f"Unsupported benchmark: {run.benchmark}")


def _aggregate_prompt_run(run: LoadedRun) -> list[AggregateRow]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for record in run.records:
        model_id = str(record["model_id"])
        backend = str(record["backend"])
        overall_key = ("overall", "all", model_id, backend)
        category_key = ("prompt_category", str(record.get("category", "unknown")), model_id, backend)
        groups.setdefault(overall_key, []).append(record)
        groups.setdefault(category_key, []).append(record)
    rows: list[AggregateRow] = []
    for (group_type, group_name, model_id, backend), records in groups.items():
        rows.append(
            AggregateRow(
                run_id=run.run_id,
                run_label=run.label,
                generated_at_utc=run.generated_at_utc,
                benchmark=run.benchmark,
                group_type=group_type,
                group_name=group_name,
                model_id=model_id,
                model_name=str(records[0]["model_name"]),
                backend=backend,
                record_count=len(records),
                avg_latency_seconds=_avg(records, "latency_seconds"),
                avg_tokens_per_second=_avg(records, "tokens_per_second"),
                avg_response_length_words=_avg(records, "response_length_words"),
                avg_rubric_score=_avg(records, "rubric_score"),
                prompt_strict_rate=None,
                prompt_loose_rate=None,
                instruction_strict_rate=None,
                instruction_loose_rate=None,
                error_rate=_mean_bool(record.get("error") for record in records),
            )
        )
    return rows


def _aggregate_ifeval_run(run: LoadedRun) -> list[AggregateRow]:
    rows: list[AggregateRow] = []
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    instruction_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}

    for record in run.records:
        model_id = str(record["model_id"])
        backend = str(record["backend"])
        groups.setdefault(("overall", "all", model_id, backend), []).append(record)
        for instruction_id, strict_ok, loose_ok in zip(
            record.get("instruction_id_list", []),
            record.get("instruction_strict_list", []),
            record.get("instruction_loose_list", []),
        ):
            instruction_type = str(instruction_id).split(":")[0]
            instruction_groups.setdefault((instruction_type, model_id, backend), []).append(
                {
                    "instruction_id": instruction_id,
                    "strict_ok": bool(strict_ok),
                    "loose_ok": bool(loose_ok),
                    "model_name": record["model_name"],
                }
            )

    for (group_type, group_name, model_id, backend), records in groups.items():
        rows.append(
            AggregateRow(
                run_id=run.run_id,
                run_label=run.label,
                generated_at_utc=run.generated_at_utc,
                benchmark=run.benchmark,
                group_type=group_type,
                group_name=group_name,
                model_id=model_id,
                model_name=str(records[0]["model_name"]),
                backend=backend,
                record_count=len(records),
                avg_latency_seconds=_avg(records, "latency_seconds"),
                avg_tokens_per_second=_avg(records, "tokens_per_second"),
                avg_response_length_words=_avg_response_words(records),
                avg_rubric_score=None,
                prompt_strict_rate=_mean_bool(record.get("prompt_strict") for record in records),
                prompt_loose_rate=_mean_bool(record.get("prompt_loose") for record in records),
                instruction_strict_rate=_safe_divide(
                    sum(int(record.get("instruction_strict_followed", 0)) for record in records),
                    sum(int(record.get("instruction_count", 0)) for record in records),
                ),
                instruction_loose_rate=_safe_divide(
                    sum(int(record.get("instruction_loose_followed", 0)) for record in records),
                    sum(int(record.get("instruction_count", 0)) for record in records),
                ),
                error_rate=_mean_bool(record.get("error") for record in records),
            )
        )

    for (instruction_type, model_id, backend), occurrences in instruction_groups.items():
        rows.append(
            AggregateRow(
                run_id=run.run_id,
                run_label=run.label,
                generated_at_utc=run.generated_at_utc,
                benchmark=run.benchmark,
                group_type="instruction_type",
                group_name=instruction_type,
                model_id=model_id,
                model_name=str(occurrences[0]["model_name"]),
                backend=backend,
                record_count=len(occurrences),
                avg_latency_seconds=None,
                avg_tokens_per_second=None,
                avg_response_length_words=None,
                avg_rubric_score=None,
                prompt_strict_rate=None,
                prompt_loose_rate=None,
                instruction_strict_rate=_mean_bool(item["strict_ok"] for item in occurrences),
                instruction_loose_rate=_mean_bool(item["loose_ok"] for item in occurrences),
                error_rate=None,
            )
        )
    return rows


def _row_key(row: AggregateRow) -> tuple[str, str, str, str, str]:
    return (row.benchmark, row.group_type, row.group_name, row.model_id, row.backend)


def _compare_aggregate_rows(baseline: AggregateRow, candidate: AggregateRow) -> dict[str, Any]:
    row = {
        "baseline_run_id": baseline.run_id,
        "candidate_run_id": candidate.run_id,
        "benchmark": baseline.benchmark,
        "group_type": baseline.group_type,
        "group_name": baseline.group_name,
        "model_id": baseline.model_id,
        "model_name": baseline.model_name,
        "backend": baseline.backend,
        "baseline_record_count": baseline.record_count,
        "candidate_record_count": candidate.record_count,
    }
    score_labels: list[str] = []
    for metric_name, direction in (
        ("avg_latency_seconds", "lower"),
        ("avg_tokens_per_second", "higher"),
        ("avg_response_length_words", "neutral"),
        ("avg_rubric_score", "higher"),
        ("prompt_strict_rate", "higher"),
        ("prompt_loose_rate", "higher"),
        ("instruction_strict_rate", "higher"),
        ("instruction_loose_rate", "higher"),
        ("error_rate", "lower"),
    ):
        base_value = getattr(baseline, metric_name)
        cand_value = getattr(candidate, metric_name)
        delta = _delta(base_value, cand_value)
        label = _metric_label(base_value, cand_value, direction)
        row[f"baseline_{metric_name}"] = base_value
        row[f"candidate_{metric_name}"] = cand_value
        row[f"delta_{metric_name}"] = delta
        row[f"label_{metric_name}"] = label
        if direction != "neutral" and label in {"improved", "regressed", "unchanged"}:
            score_labels.append(label)

    improved = score_labels.count("improved")
    regressed = score_labels.count("regressed")
    if improved > regressed:
        row["overall_label"] = "improved"
    elif regressed > improved:
        row["overall_label"] = "regressed"
    else:
        row["overall_label"] = "unchanged"
    return row


def _delta(base_value: float | None, candidate_value: float | None) -> float | None:
    if base_value is None or candidate_value is None:
        return None
    return round(candidate_value - base_value, 6)


def _metric_label(base_value: float | None, candidate_value: float | None, direction: str) -> str:
    if base_value is None or candidate_value is None:
        return "n/a"
    if abs(candidate_value - base_value) < 1e-9:
        return "unchanged"
    if direction == "higher":
        return "improved" if candidate_value > base_value else "regressed"
    if direction == "lower":
        return "improved" if candidate_value < base_value else "regressed"
    return "changed"


def _build_report(baseline: LoadedRun, candidates: list[LoadedRun], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Comparison Report",
        "",
        f"- Baseline: `{baseline.label}` ({baseline.benchmark})",
    ]
    for candidate in candidates:
        lines.append(f"- Compared run: `{candidate.label}` ({candidate.benchmark})")
    lines.append("")

    if not rows:
        lines.extend(["No shared comparison slices were found between the selected runs.", ""])
        return "\n".join(lines)

    lines.extend(["## Summary", ""])
    for candidate in candidates:
        candidate_rows = [row for row in rows if row["candidate_run_id"] == candidate.run_id]
        improved = sum(1 for row in candidate_rows if row["overall_label"] == "improved")
        regressed = sum(1 for row in candidate_rows if row["overall_label"] == "regressed")
        unchanged = sum(1 for row in candidate_rows if row["overall_label"] == "unchanged")
        lines.append(
            f"- `{candidate.label}`: improved `{improved}`, regressed `{regressed}`, unchanged `{unchanged}`"
        )
    lines.append("")

    sorted_rows = sorted(
        rows,
        key=lambda row: (
            0 if row["overall_label"] == "regressed" else 1,
            abs(row.get("delta_avg_rubric_score") or row.get("delta_prompt_strict_rate") or 0),
        ),
        reverse=True,
    )
    lines.extend(["## Representative Deltas", "", "| Candidate | Slice | Model | Overall | Latency Δ | TPS Δ | Score Δ |", "| --- | --- | --- | --- | ---: | ---: | ---: |"])
    for row in sorted_rows[:10]:
        score_delta = row.get("delta_avg_rubric_score")
        if score_delta is None:
            score_delta = row.get("delta_prompt_strict_rate")
        lines.append(
            f"| {row['candidate_run_id']} | {row['group_type']}:{row['group_name']} | "
            f"{row['model_id']} | {row['overall_label']} | "
            f"{_fmt(row.get('delta_avg_latency_seconds'))} | "
            f"{_fmt(row.get('delta_avg_tokens_per_second'))} | "
            f"{_fmt(score_delta)} |"
        )
    lines.append("")
    return "\n".join(lines)


def _avg(records: list[dict[str, Any]], key: str) -> float | None:
    values = [float(record[key]) for record in records if record.get(key) is not None]
    if not values:
        return None
    return round(mean(values), 6)


def _avg_response_words(records: list[dict[str, Any]]) -> float | None:
    values = [len(str(record.get("response_text", "")).split()) for record in records]
    if not values:
        return None
    return round(mean(values), 6)


def _safe_divide(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 6)


def _mean_bool(values: Any) -> float | None:
    normalized = [bool(value) for value in values if value is not None]
    if not normalized:
        return None
    return round(sum(int(value) for value in normalized) / len(normalized), 6)


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"

