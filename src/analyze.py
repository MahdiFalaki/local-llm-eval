"""Analysis layer for completed evaluation runs."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from src.reporting import LoadedRun, load_run, write_csv, write_markdown
from src.utils import ensure_directory, shorten, write_json


def analyze_run(run_path: Path, output_dir: Path) -> Path:
    """Analyze one completed run and write offline artifacts."""
    run = load_run(run_path)
    output_dir = ensure_directory(output_dir)

    failure_rows = [_build_failure_row(run, record) for record in run.records]
    analysis_rows = _build_analysis_rows(run, failure_rows)
    model_summary_rows = _build_model_summary_rows(run, analysis_rows)
    failure_breakdown_rows = _build_failure_breakdown_rows(run, failure_rows)
    summary = _build_summary(run, failure_rows, analysis_rows)

    write_json(output_dir / "analysis_summary.json", summary)
    write_csv(output_dir / "analysis_by_category.csv", analysis_rows)
    write_csv(output_dir / "analysis_failures.csv", failure_rows)
    write_csv(output_dir / "model_summary.csv", model_summary_rows)
    write_csv(output_dir / "failure_breakdown.csv", failure_breakdown_rows)
    write_markdown(output_dir / "model_summary.md", _build_model_summary_markdown(run, model_summary_rows))
    write_markdown(
        output_dir / "failure_breakdown.md",
        _build_failure_breakdown_markdown(run, failure_breakdown_rows),
    )
    write_markdown(
        output_dir / "analysis_report.md",
        _build_report(run, summary, analysis_rows, failure_rows, model_summary_rows, failure_breakdown_rows),
    )
    return output_dir


def _build_failure_row(run: LoadedRun, record: dict[str, Any]) -> dict[str, Any]:
    failure_types = _failure_types(run.benchmark, record)
    prompt_text = str(record.get("prompt") or record.get("prompt_id") or "")
    context_bucket = _context_bucket(prompt_text) if prompt_text else "unknown"
    return {
        "benchmark": run.benchmark,
        "model_id": str(record.get("model_id", "")),
        "model_name": str(record.get("model_name", "")),
        "backend": str(record.get("backend", "")),
        "temperature": record.get("temperature"),
        "group_key": record.get("category") or record.get("example_key") or record.get("prompt_id"),
        "context_length_bucket": context_bucket,
        "failure_types": ";".join(failure_types),
        "has_failure": bool(failure_types),
        "error": record.get("error"),
        "response_preview": shorten(str(record.get("response_text", ""))),
        "prompt_preview": shorten(prompt_text),
        "avg_response_length_words": _response_length_words(record),
        "prompt_strict": record.get("prompt_strict"),
        "prompt_loose": record.get("prompt_loose"),
        "rubric_score": record.get("rubric_score"),
    }


def _failure_types(benchmark: str, record: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    response_text = str(record.get("response_text", "")).strip()
    error = str(record.get("error") or "")
    output_tokens = int(record.get("output_tokens") or 0)
    max_tokens = int(record.get("max_tokens") or 0)

    if error:
        failures.append("timeout_or_backend_error" if "timed out" in error.lower() else "backend_error")
    if not response_text:
        failures.append("empty_output")
    if output_tokens and max_tokens and output_tokens >= int(max_tokens * 0.95):
        failures.append("truncation_suspected")
    if output_tokens and output_tokens <= 10:
        failures.append("too_short_output")

    if benchmark == "prompts":
        if record.get("rubric_format_compliance") is False:
            failures.append("format_compliance_failure")
        keyword_total = int(record.get("rubric_keyword_total") or 0)
        keyword_hits = int(record.get("rubric_keyword_hits") or 0)
        if keyword_total > 0 and keyword_hits < keyword_total:
            failures.append("keyword_coverage_failure")
    elif benchmark == "ifeval":
        if record.get("prompt_strict") is False or record.get("instruction_strict_followed", 0) < record.get("instruction_count", 0):
            failures.append("instruction_check_failure")

    seen: set[str] = set()
    ordered: list[str] = []
    for item in failures:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def _build_analysis_rows(run: LoadedRun, failure_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base_records = list(zip(run.records, failure_rows))
    grouped: dict[tuple[str, str, str, str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)

    for record, failure in base_records:
        model_id = str(record.get("model_id", ""))
        backend = str(record.get("backend", ""))
        temperature = str(record.get("temperature", ""))
        grouped[("overall", "all", model_id, backend, temperature)].append((record, failure))
        grouped[("temperature", temperature, model_id, backend, temperature)].append((record, failure))
        if failure["context_length_bucket"] != "unknown":
            grouped[("context_length_bucket", failure["context_length_bucket"], model_id, backend, temperature)].append((record, failure))
        if run.benchmark == "prompts":
            grouped[("prompt_category", str(record.get("category", "unknown")), model_id, backend, temperature)].append((record, failure))
        else:
            for instruction_id in record.get("instruction_id_list", []):
                instruction_type = str(instruction_id).split(":")[0]
                grouped[("instruction_type", instruction_type, model_id, backend, temperature)].append((record, failure))

    for (dimension, value, model_id, backend, temperature), items in grouped.items():
        records = [record for record, _ in items]
        failures = [failure for _, failure in items]
        rows.append(
            {
                "benchmark": run.benchmark,
                "dimension": dimension,
                "group_value": value,
                "model_id": model_id,
                "model_name": str(records[0].get("model_name", "")),
                "backend": backend,
                "temperature": temperature,
                "record_count": len(records),
                "failure_rate": _mean_bool(failure["has_failure"] for failure in failures),
                "avg_latency_seconds": _avg(records, "latency_seconds"),
                "avg_tokens_per_second": _avg(records, "tokens_per_second"),
                "avg_response_length_words": round(mean(_response_length_words(record) for record in records), 6),
                "avg_rubric_score": _avg(records, "rubric_score"),
                "prompt_strict_rate": _mean_bool(record.get("prompt_strict") for record in records),
                "prompt_loose_rate": _mean_bool(record.get("prompt_loose") for record in records),
                "instruction_strict_rate": _safe_divide(
                    sum(int(record.get("instruction_strict_followed", 0)) for record in records),
                    sum(int(record.get("instruction_count", 0)) for record in records),
                ),
                "instruction_loose_rate": _safe_divide(
                    sum(int(record.get("instruction_loose_followed", 0)) for record in records),
                    sum(int(record.get("instruction_count", 0)) for record in records),
                ),
                "error_rate": _mean_bool(record.get("error") for record in records),
            }
        )

    rows.sort(key=lambda row: (row["dimension"], row["group_value"], row["model_id"]))
    return rows


def _build_summary(run: LoadedRun, failure_rows: list[dict[str, Any]], analysis_rows: list[dict[str, Any]]) -> dict[str, Any]:
    failure_counter = Counter()
    for row in failure_rows:
        for failure_type in filter(None, row["failure_types"].split(";")):
            failure_counter[failure_type] += 1

    overall_rows = [row for row in analysis_rows if row["dimension"] == "overall"]
    primary_metric = "avg_rubric_score" if run.benchmark == "prompts" else "prompt_strict_rate"
    best_row = max(overall_rows, key=lambda row: row.get(primary_metric) or float("-inf")) if overall_rows else None
    worst_row = min(overall_rows, key=lambda row: row.get(primary_metric) or float("inf")) if overall_rows else None

    return {
        "run": {
            "run_id": run.run_id,
            "label": run.label,
            "benchmark": run.benchmark,
            "generated_at_utc": run.generated_at_utc,
            "path": str(run.path),
        },
        "record_count": len(run.records),
        "failure_type_totals": dict(failure_counter.most_common()),
        "best_model_snapshot": best_row,
        "worst_model_snapshot": worst_row,
    }


def _build_model_summary_rows(run: LoadedRun, analysis_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [row for row in analysis_rows if row["dimension"] == "overall"]
    rows.sort(key=lambda row: row["model_id"])
    model_summary_rows: list[dict[str, Any]] = []
    for row in rows:
        primary_strict = row.get("prompt_strict_rate")
        primary_loose = row.get("prompt_loose_rate")
        if run.benchmark == "prompts":
            primary_strict = row.get("avg_rubric_score")
            primary_loose = None
        best_category, worst_category = _best_and_worst_category(run, analysis_rows, row["model_id"])
        model_summary_rows.append(
            {
                "model_id": row["model_id"],
                "model_name": row["model_name"],
                "backend": row["backend"],
                "record_count": row["record_count"],
                "strict_score_or_pass_rate": primary_strict if primary_strict is not None else "n/a",
                "loose_score_or_pass_rate": primary_loose if primary_loose is not None else "n/a",
                "instruction_strict_rate": row.get("instruction_strict_rate", "n/a"),
                "instruction_loose_rate": row.get("instruction_loose_rate", "n/a"),
                "error_rate": row.get("error_rate", "n/a"),
                "average_latency_seconds": row.get("avg_latency_seconds", "n/a"),
                "average_tokens_per_second": row.get("avg_tokens_per_second", "n/a"),
                "average_response_length_words": row.get("avg_response_length_words", "n/a"),
                "best_category": best_category,
                "worst_category": worst_category,
            }
        )
    return model_summary_rows


def _build_failure_breakdown_rows(run: LoadedRun, failure_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row, record in zip(failure_rows, run.records):
        model_id = str(row["model_id"])
        backend = str(row["backend"])
        grouped[(model_id, row["model_name"], backend, "overall")].append(row)
        if run.benchmark == "ifeval":
            instruction_types = {str(instruction_id).split(":")[0] for instruction_id in record.get("instruction_id_list", [])}
            for instruction_type in instruction_types:
                grouped[(model_id, row["model_name"], backend, instruction_type)].append(row)
        else:
            grouped[(model_id, row["model_name"], backend, str(record.get("category", "unknown")))].append(row)

    breakdown_rows: list[dict[str, Any]] = []
    for (model_id, model_name, backend, category), rows in sorted(grouped.items()):
        failure_counter = Counter()
        fail_count = 0
        for row in rows:
            failure_types = [item for item in row["failure_types"].split(";") if item]
            if failure_types:
                fail_count += 1
                failure_counter.update(failure_types)
        record_count = len(rows)
        breakdown_rows.append(
            {
                "model_id": model_id,
                "model_name": model_name,
                "backend": backend,
                "scope": "overall" if category == "overall" else "category",
                "instruction_or_prompt_category": category,
                "record_count": record_count,
                "fail_count": fail_count,
                "fail_rate": round(fail_count / record_count, 6) if record_count else 0.0,
                "dominant_failure_type": failure_counter.most_common(1)[0][0] if failure_counter else "none",
            }
        )
    breakdown_rows.sort(
        key=lambda row: (
            row["model_id"],
            0 if row["scope"] == "overall" else 1,
            -int(row["fail_count"]),
            -float(row["fail_rate"]),
            str(row["instruction_or_prompt_category"]),
        )
    )
    return breakdown_rows


def _build_report(
    run: LoadedRun,
    summary: dict[str, Any],
    analysis_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
    model_summary_rows: list[dict[str, Any]],
    failure_breakdown_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Analysis Report",
        "",
        f"- Run: `{run.label}`",
        f"- Benchmark: `{run.benchmark}`",
        f"- Records: `{len(run.records)}`",
        "",
        "## High-level Summary",
        "",
    ]

    best = summary.get("best_model_snapshot")
    worst = summary.get("worst_model_snapshot")
    if best:
        lines.append(
            f"- Best model overall: `{best['model_id']}` with strict score/pass rate "
            f"`{_fmt(best.get('avg_rubric_score') or best.get('prompt_strict_rate'))}`"
        )
    if worst:
        lines.append(
            f"- Weakest model overall: `{worst['model_id']}` with strict score/pass rate "
            f"`{_fmt(worst.get('avg_rubric_score') or worst.get('prompt_strict_rate'))}`"
        )
    if not best and not worst:
        lines.append("- No aggregate model snapshots were available.")
    lines.append("")

    lines.extend(["## Model Summary", ""])
    lines.extend(_markdown_table(
        [
            "Model",
            "Backend",
            "Strict",
            "Loose",
            "Error Rate",
            "Avg Latency",
            "Avg TPS",
            "Best Category",
            "Worst Category",
        ],
        [
            [
                row["model_id"],
                row["backend"],
                _fmt_or_na(row["strict_score_or_pass_rate"]),
                _fmt_or_na(row["loose_score_or_pass_rate"]),
                _fmt_or_na(row["error_rate"]),
                _fmt_or_na(row["average_latency_seconds"]),
                _fmt_or_na(row["average_tokens_per_second"]),
                row["best_category"],
                row["worst_category"],
            ]
            for row in model_summary_rows
        ],
    ))
    lines.append("")

    lines.extend(["## Most Common Failure Types", "", "| Failure Type | Count |", "| --- | ---: |"])
    for failure_type, count in summary.get("failure_type_totals", {}).items():
        lines.append(f"| {failure_type} | {count} |")
    if not summary.get("failure_type_totals"):
        lines.append("| none | 0 |")
    lines.append("")

    lines.extend(["## Failure Breakdown", ""])
    overall_breakdown = [row for row in failure_breakdown_rows if row["scope"] == "overall"]
    category_breakdown = [row for row in failure_breakdown_rows if row["scope"] == "category"]
    lines.extend(["### Overall by Model", ""])
    lines.extend(_markdown_table(
        [
            "Model",
            "Fail Count",
            "Fail Rate",
            "Dominant Failure",
        ],
        [
            [
                row["model_id"],
                str(row["fail_count"]),
                _fmt_or_na(row["fail_rate"]),
                row["dominant_failure_type"],
            ]
            for row in overall_breakdown
        ],
    ))
    lines.append("")
    lines.extend(["### By Category", ""])
    lines.extend(_markdown_table(
        [
            "Model",
            "Category",
            "Fail Count",
            "Fail Rate",
            "Dominant Failure",
        ],
        [
            [
                row["model_id"],
                row["instruction_or_prompt_category"],
                str(row["fail_count"]),
                _fmt_or_na(row["fail_rate"]),
                row["dominant_failure_type"],
            ]
            for row in category_breakdown[:18]
        ],
    ))
    lines.append("")

    lines.extend(["## Strengths and Weaknesses", ""])
    for model_row in model_summary_rows:
        slice_rows = [
            row for row in analysis_rows
            if row["model_id"] == model_row["model_id"] and row["dimension"] in {"instruction_type", "prompt_category"}
        ]
        if not slice_rows:
            continue
        primary_key = "prompt_strict_rate" if run.benchmark == "ifeval" else "avg_rubric_score"
        strongest = max(slice_rows, key=lambda row: row.get(primary_key) if row.get(primary_key) is not None else float("-inf"))
        weakest = min(slice_rows, key=lambda row: row.get(primary_key) if row.get(primary_key) is not None else float("inf"))
        lines.append(
            f"- `{model_row['model_id']}` strongest on `{strongest['group_value']}` "
            f"with score `{_fmt(strongest.get(primary_key))}`; weakest on `{weakest['group_value']}` "
            f"with score `{_fmt(weakest.get(primary_key))}`."
        )
    lines.append("")

    failed_examples = [row for row in failure_rows if row["has_failure"]][:5]
    lines.extend(["## Representative Failed Examples", ""])
    if failed_examples:
        for row in failed_examples:
            lines.append(
                f"- `{row['model_id']}` failures=`{row['failure_types']}` "
                f"prompt=`{row['prompt_preview']}` response=`{row['response_preview']}`"
            )
    else:
        lines.append("- No failed examples were identified.")
    lines.append("")
    return "\n".join(lines)


def _build_model_summary_markdown(run: LoadedRun, rows: list[dict[str, Any]]) -> str:
    title = "# Model Summary\n\n"
    title += f"- Run: `{run.label}`\n- Benchmark: `{run.benchmark}`\n\n"
    table = _markdown_table(
        [
            "Model",
            "Backend",
            "Strict",
            "Loose",
            "Instr. Strict",
            "Instr. Loose",
            "Error Rate",
            "Avg Latency",
            "Avg TPS",
            "Best Category",
            "Worst Category",
        ],
        [
            [
                row["model_id"],
                row["backend"],
                _fmt_or_na(row["strict_score_or_pass_rate"]),
                _fmt_or_na(row["loose_score_or_pass_rate"]),
                _fmt_or_na(row["instruction_strict_rate"]),
                _fmt_or_na(row["instruction_loose_rate"]),
                _fmt_or_na(row["error_rate"]),
                _fmt_or_na(row["average_latency_seconds"]),
                _fmt_or_na(row["average_tokens_per_second"]),
                row["best_category"],
                row["worst_category"],
            ]
            for row in rows
        ],
    )
    return title + "\n".join(table) + "\n"


def _build_failure_breakdown_markdown(run: LoadedRun, rows: list[dict[str, Any]]) -> str:
    title = "# Failure Breakdown\n\n"
    title += f"- Run: `{run.label}`\n- Benchmark: `{run.benchmark}`\n\n"
    overall_rows = [row for row in rows if row["scope"] == "overall"]
    category_rows = [row for row in rows if row["scope"] == "category"]
    overall_table = _markdown_table(
        [
            "Model",
            "Fail Count",
            "Fail Rate",
            "Dominant Failure",
        ],
        [
            [
                row["model_id"],
                str(row["fail_count"]),
                _fmt_or_na(row["fail_rate"]),
                row["dominant_failure_type"],
            ]
            for row in overall_rows
        ],
    )
    category_table = _markdown_table(
        [
            "Model",
            "Category",
            "Fail Count",
            "Fail Rate",
            "Dominant Failure",
        ],
        [
            [
                row["model_id"],
                row["instruction_or_prompt_category"],
                str(row["fail_count"]),
                _fmt_or_na(row["fail_rate"]),
                row["dominant_failure_type"],
            ]
            for row in category_rows
        ],
    )
    return (
        title
        + "## Overall by Model\n\n"
        + "\n".join(overall_table)
        + "\n\n## By Category\n\n"
        + "\n".join(category_table)
        + "\n"
    )


def _best_and_worst_category(
    run: LoadedRun,
    analysis_rows: list[dict[str, Any]],
    model_id: str,
) -> tuple[str, str]:
    relevant_dimensions = {"instruction_type"} if run.benchmark == "ifeval" else {"prompt_category"}
    slice_rows = [
        row
        for row in analysis_rows
        if row["model_id"] == model_id and row["dimension"] in relevant_dimensions
    ]
    if not slice_rows:
        return "n/a", "n/a"
    primary_key = "prompt_strict_rate" if run.benchmark == "ifeval" else "avg_rubric_score"
    strongest = max(
        slice_rows,
        key=lambda row: row.get(primary_key) if row.get(primary_key) is not None else float("-inf"),
    )
    weakest = min(
        slice_rows,
        key=lambda row: row.get(primary_key) if row.get(primary_key) is not None else float("inf"),
    )
    return str(strongest["group_value"]), str(weakest["group_value"])


def _response_length_words(record: dict[str, Any]) -> int:
    if record.get("response_length_words") is not None:
        return int(record["response_length_words"])
    return len(str(record.get("response_text", "")).split())


def _context_bucket(prompt_text: str) -> str:
    word_count = len(prompt_text.split())
    if word_count < 50:
        return "short"
    if word_count < 150:
        return "medium"
    return "long"


def _avg(records: list[dict[str, Any]], key: str) -> float | None:
    values = [float(record[key]) for record in records if record.get(key) is not None]
    if not values:
        return None
    return round(mean(values), 6)


def _mean_bool(values: Any) -> float | None:
    normalized = [bool(value) for value in values if value is not None]
    if not normalized:
        return None
    return round(sum(int(value) for value in normalized) / len(normalized), 6)


def _safe_divide(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 6)


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _fmt_or_na(value: Any) -> str:
    if value == "n/a" or value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return str(value)


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines
