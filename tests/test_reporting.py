from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.reporting import (
    default_analysis_output_dir,
    load_run,
    resolve_run_path,
    save_prompt_run_snapshot,
)

from tests.helpers import create_ifeval_run, create_prompt_run


class ReportingTests(unittest.TestCase):
    def test_load_prompt_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            results_path = create_prompt_run(Path(tmp_dir) / "prompts")
            run = load_run(results_path)
        self.assertEqual(run.benchmark, "prompts")
        self.assertEqual(len(run.records), 2)

    def test_load_ifeval_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = create_ifeval_run(Path(tmp_dir) / "ifeval-full")
            run = load_run(run_dir)
        self.assertEqual(run.benchmark, "ifeval")
        self.assertEqual(run.label, "ifeval-full")
        self.assertEqual(len(run.records), 3)

    def test_load_run_raises_for_malformed_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            bad_path = Path(tmp_dir) / "results.json"
            bad_path.write_text("{broken", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_run(bad_path)

    def test_resolve_run_path_raises_for_unknown_path(self) -> None:
        with self.assertRaises(FileNotFoundError):
            resolve_run_path(Path("/definitely/not/here"))

    def test_save_prompt_run_snapshot_and_default_analysis_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir) / "outputs"
            run_dir = save_prompt_run_snapshot(
                output_root=output_root,
                run_name="Prompt Run",
                records=[{"model_id": "qwen2_5_7b", "prompt_id": "p1"}],
                environment={"ok": True},
                config_snapshot={"benchmark": "prompts"},
            )
            run = load_run(run_dir / "results.json")
            analysis_dir = default_analysis_output_dir(output_root, run)
            self.assertTrue((run_dir / "results.csv").exists())
            self.assertEqual(analysis_dir, run_dir / "analysis")
