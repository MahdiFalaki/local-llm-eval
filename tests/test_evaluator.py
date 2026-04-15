from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.evaluator import load_model_configs, load_prompt_configs


class EvaluatorConfigTests(unittest.TestCase):
    def test_load_model_configs_filters_enabled_backend_and_selection(self) -> None:
        yaml_text = """
defaults:
  timeout: 99
models:
  - id: qwen2_5_7b
    backend: ollama
    model_name: qwen2.5:7b
  - id: llama3_1_8b
    backend: ollama
    model_name: llama3.1:8b
    enabled: false
  - id: vllm_model
    backend: vllm
    model_name: local-vllm
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "models.yaml"
            path.write_text(yaml_text, encoding="utf-8")
            configs = load_model_configs(path, backend_filter="ollama", selected_ids=["qwen2_5_7b"])
        self.assertEqual([config.id for config in configs], ["qwen2_5_7b"])
        self.assertEqual(configs[0].timeout, 99)

    def test_load_model_configs_raises_when_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "models.yaml"
            path.write_text("models: []\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_model_configs(path)

    def test_load_prompt_configs_uses_defaults(self) -> None:
        yaml_text = """
prompts:
  - id: p1
    category: summarization
    prompt: Summarize this
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "prompts.yaml"
            path.write_text(yaml_text, encoding="utf-8")
            prompts = load_prompt_configs(path)
        self.assertEqual(prompts[0].temperatures, [0.2])
        self.assertEqual(prompts[0].max_tokens, 128)
        self.assertEqual(prompts[0].expected_keywords, [])

