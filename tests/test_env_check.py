from __future__ import annotations

import builtins
import io
import types
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock, patch

from src.env_check import detect_ollama, detect_torch_cuda, print_environment_summary


class EnvCheckTests(unittest.TestCase):
    def test_detect_torch_cuda_handles_import_failure(self) -> None:
        original_import = builtins.__import__

        def fake_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "torch":
                raise RuntimeError("broken torch")
            return original_import(name, *args, **kwargs)

        with patch("importlib.util.find_spec", return_value=object()), patch("builtins.__import__", side_effect=fake_import):
            result = detect_torch_cuda()

        self.assertTrue(result["installed"])
        self.assertFalse(result["cuda_available"])
        self.assertIn("error", result)

    def test_detect_torch_cuda_uses_fake_module(self) -> None:
        fake_torch = types.SimpleNamespace(
            __version__="1.0",
            cuda=types.SimpleNamespace(
                is_available=lambda: True,
                device_count=lambda: 1,
                get_device_properties=lambda index: types.SimpleNamespace(name="GPU", total_memory=8 * 1024 ** 3),
            ),
        )
        with patch("importlib.util.find_spec", return_value=object()), patch.dict("sys.modules", {"torch": fake_torch}):
            result = detect_torch_cuda()
        self.assertEqual(result["gpu_count"], 1)
        self.assertEqual(result["gpus"][0]["name"], "GPU")

    def test_detect_ollama_reports_models(self) -> None:
        response = Mock(ok=True)
        response.json.return_value = {"models": [{"name": "qwen"}]}
        with patch("shutil.which", return_value="/usr/bin/ollama"), patch("requests.get", return_value=response):
            status = detect_ollama("http://localhost:11434")
        self.assertTrue(status.installed)
        self.assertTrue(status.reachable)
        self.assertEqual(status.details["models_detected"], 1)

    def test_print_environment_summary(self) -> None:
        summary = {
            "runtime": {"os_name": "Linux", "os_version": "Linux-1", "python_version": "3.11", "python_executable": "/usr/bin/python"},
            "hardware": {"cpu_name": "x86_64", "cpu_count_logical": 8, "cpu_count_physical": 4, "ram_gb": 16.0},
            "torch_cuda": {"installed": False, "cuda_available": False, "gpu_count": 0, "gpus": []},
            "nvidia_smi": {"installed": False, "gpu_count": 0, "gpus": []},
            "backends": {"ollama": {"installed": True, "reachable": True, "details": {}}, "vllm": {"installed": False, "reachable": False, "details": {}}},
        }
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            print_environment_summary(summary)
        output = buffer.getvalue()
        self.assertIn("Environment Summary", output)
        self.assertIn("ollama: installed=True", output)
