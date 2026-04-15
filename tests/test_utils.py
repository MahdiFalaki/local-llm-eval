from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.utils import build_prompt_text, estimate_token_count, load_yaml, sanitize_name, shorten


class UtilsTests(unittest.TestCase):
    def test_load_yaml_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "config.yaml"
            path.write_text("models:\n  - id: demo\n", encoding="utf-8")
            payload = load_yaml(path)
        self.assertEqual(payload["models"][0]["id"], "demo")

    def test_load_yaml_rejects_non_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "config.yaml"
            path.write_text("- item\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_yaml(path)

    def test_text_helpers(self) -> None:
        self.assertEqual(build_prompt_text(" Prompt "), "Prompt")
        self.assertEqual(build_prompt_text("Prompt", " Context "), "Prompt\n\nContext:\nContext")
        self.assertEqual(sanitize_name("  demo run / test "), "demo-run-test")
        self.assertEqual(shorten("a b c", limit=10), "a b c")
        self.assertTrue(shorten("one two three four five", limit=10).endswith("..."))
        self.assertEqual(estimate_token_count(""), 0)
        self.assertGreaterEqual(estimate_token_count("one two three"), 1)
