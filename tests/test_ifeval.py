from __future__ import annotations

import types
import unittest
from unittest.mock import patch

from src.ifeval import dataset_summary, load_ifeval_examples


class IFEvalTests(unittest.TestCase):
    def test_load_ifeval_examples_raises_clear_error_without_datasets(self) -> None:
        with patch("importlib.import_module", side_effect=ImportError("missing datasets")):
            with self.assertRaises(RuntimeError) as exc:
                load_ifeval_examples(limit=1)
        self.assertIn("datasets", str(exc.exception))

    def test_load_ifeval_examples_uses_lazy_loader(self) -> None:
        fake_module = types.SimpleNamespace(
            load_dataset=lambda name, split: [
                {
                    "key": 1,
                    "prompt": "prompt",
                    "instruction_id_list": ["keywords:include"],
                    "kwargs": [{"keyword": "alpha"}],
                }
            ]
        )
        with patch("importlib.import_module", return_value=fake_module):
            examples = load_ifeval_examples(limit=1)
        self.assertEqual(examples[0].key, 1)
        self.assertEqual(examples[0].instruction_id_list, ["keywords:include"])

    def test_dataset_summary(self) -> None:
        examples = [
            types.SimpleNamespace(instruction_id_list=["keywords:include"], key=1),
            types.SimpleNamespace(instruction_id_list=["language:en", "keywords:include"], key=2),
        ]
        summary = dataset_summary(examples)
        self.assertEqual(summary["example_count"], 2)
        self.assertEqual(summary["instruction_type_totals"]["keywords"], 2)
        self.assertEqual(summary["instructions_per_prompt"][1], 1)
