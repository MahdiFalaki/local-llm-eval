from __future__ import annotations

import unittest

from src.compare import AggregateRow, _compare_aggregate_rows, _metric_label


class CompareTests(unittest.TestCase):
    def test_metric_label(self) -> None:
        self.assertEqual(_metric_label(1.0, 2.0, "higher"), "improved")
        self.assertEqual(_metric_label(2.0, 1.0, "lower"), "improved")
        self.assertEqual(_metric_label(1.0, 1.0, "higher"), "unchanged")
        self.assertEqual(_metric_label(None, 1.0, "higher"), "n/a")

    def test_compare_rows_overall_label(self) -> None:
        baseline = AggregateRow(
            run_id="base",
            run_label="base",
            generated_at_utc="2026-04-15T00:00:00+00:00",
            benchmark="ifeval",
            group_type="overall",
            group_name="all",
            model_id="qwen2_5_7b",
            model_name="qwen2.5:7b",
            backend="ollama",
            record_count=10,
            avg_latency_seconds=2.0,
            avg_tokens_per_second=100.0,
            avg_response_length_words=100.0,
            avg_rubric_score=None,
            prompt_strict_rate=0.7,
            prompt_loose_rate=0.8,
            instruction_strict_rate=0.75,
            instruction_loose_rate=0.8,
            error_rate=0.1,
        )
        candidate = AggregateRow(
            run_id="cand",
            run_label="cand",
            generated_at_utc="2026-04-15T01:00:00+00:00",
            benchmark="ifeval",
            group_type="overall",
            group_name="all",
            model_id="qwen2_5_7b",
            model_name="qwen2.5:7b",
            backend="ollama",
            record_count=10,
            avg_latency_seconds=1.5,
            avg_tokens_per_second=110.0,
            avg_response_length_words=100.0,
            avg_rubric_score=None,
            prompt_strict_rate=0.72,
            prompt_loose_rate=0.82,
            instruction_strict_rate=0.8,
            instruction_loose_rate=0.85,
            error_rate=0.05,
        )
        row = _compare_aggregate_rows(baseline, candidate)
        self.assertEqual(row["overall_label"], "improved")
        self.assertGreater(row["delta_prompt_strict_rate"], 0)

