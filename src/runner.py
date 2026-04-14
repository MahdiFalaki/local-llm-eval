"""CLI entrypoint for local LLM evaluation."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.env_check import build_environment_summary, print_environment_summary
from src.evaluator import Evaluator, load_model_configs, load_prompt_configs
from src.utils import PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description="Evaluate local LLMs with Ollama or vLLM.")
    parser.add_argument("--backend", choices=["ollama", "vllm"], default="ollama")
    parser.add_argument("--models-config", type=Path, default=PROJECT_ROOT / "configs" / "models.yaml")
    parser.add_argument("--prompts-config", type=Path, default=PROJECT_ROOT / "configs" / "prompts.yaml")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--vllm-url", default="http://localhost:8000")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    """Initialize application logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


def validate_backend_availability(summary: dict, backend: str) -> None:
    """Fail early with a clear message if the selected backend is not usable."""
    status = summary["backends"][backend]
    if backend == "ollama":
        if not status["installed"]:
            raise RuntimeError("Ollama CLI is not installed. Install Ollama or use --backend vllm.")
        if not status["reachable"]:
            raise RuntimeError(
                f"Ollama is installed but not reachable at {status['details'].get('base_url')}. "
                "Start the Ollama service and retry."
            )
        return

    if backend == "vllm":
        if not status["installed"]:
            raise RuntimeError(
                "vLLM is not installed in the current environment. "
                "Install vllm only if you want the optional backend."
            )
        if not status["reachable"]:
            raise RuntimeError(
                f"vLLM is installed but no server responded at {status['details'].get('base_url')}. "
                "Start a vLLM server and retry."
            )


def main() -> int:
    """Run the CLI."""
    args = parse_args()
    configure_logging(args.verbose)

    summary = build_environment_summary(ollama_url=args.ollama_url, vllm_url=args.vllm_url)
    print_environment_summary(summary)

    if args.check_only:
        return 0

    validate_backend_availability(summary, args.backend)
    models = load_model_configs(args.models_config, backend_filter=args.backend)
    prompts = load_prompt_configs(args.prompts_config)

    evaluator = Evaluator(args.output_dir)
    records = evaluator.run(
        models=models,
        prompts=prompts,
        backend_urls={"ollama": args.ollama_url, "vllm": args.vllm_url},
    )
    csv_path, json_path = evaluator.save(
        records=records,
        environment=summary,
        models_path=args.models_config,
        prompts_path=args.prompts_config,
    )
    print(f"Saved {len(records)} records to {csv_path}")
    print(f"Saved JSON output to {json_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
