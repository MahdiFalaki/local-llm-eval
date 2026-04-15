from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.analyze import _build_analysis_rows, _build_failure_row, _build_model_summary_rows, _failure_types, analyze_run
from src.reporting import load_run

from tests.helpers import create_ifeval_run, create_prompt_run


class AnalyzeTests(unittest.TestCase):
    def test_failure_types_for_prompt_and_ifeval(self) -> None:
        prompt_failures = _failure_types(
            "prompts",
            {
                "response_text": "",
                "error": "backend broke",
                "output_tokens": 5,
                "max_tokens": 10,
                "rubric_format_compliance": False,
                "rubric_keyword_total": 2,
                "rubric_keyword_hits": 1,
            },
        )
        ifeval_failures = _failure_types(
            "ifeval",
            {
                "response_text": "tiny",
                "error": None,
                "output_tokens": 5,
                "max_tokens": 128,
                "prompt_strict": False,
                "instruction_strict_followed": 0,
                "instruction_count": 1,
            },
        )
        self.assertIn("backend_error", prompt_failures)
        self.assertIn("empty_output", prompt_failures)
        self.assertIn("format_compliance_failure", prompt_failures)
        self.assertIn("keyword_coverage_failure", prompt_failures)
        self.assertIn("instruction_check_failure", ifeval_failures)

    def test_analyze_run_writes_expected_ifeval_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = create_ifeval_run(Path(tmp_dir) / "ifeval-full")
            output_dir = Path(tmp_dir) / "analysis"
            analyze_run(run_dir, output_dir)
            run = load_run(run_dir)
            failure_rows = [_build_failure_row(run, record) for record in run.records]
            analysis_rows = _build_analysis_rows(run, failure_rows)
            model_rows = _build_model_summary_rows(run, analysis_rows)
            self.assertTrue((output_dir / "analysis_report.md").exists())
            self.assertTrue((output_dir / "model_summary.csv").exists())
            self.assertTrue((output_dir / "failure_breakdown.md").exists())
            self.assertEqual(model_rows[0]["best_category"], "detectable_format")
            self.assertEqual(model_rows[0]["worst_category"], "keywords")

    def test_analyze_run_supports_prompt_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            results_json = create_prompt_run(Path(tmp_dir) / "prompts")
            output_dir = Path(tmp_dir) / "analysis"
            analyze_run(results_json, output_dir)
            self.assertTrue((output_dir / "analysis_summary.json").exists())
            self.assertTrue((output_dir / "analysis_failures.csv").exists())
