"""CLI entrypoint for local LLM evaluation."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import asdict
from pathlib import Path

from src.analyze import analyze_run
from src.compare import compare_runs
from src.env_check import build_environment_summary, print_environment_summary
from src.ifeval import IFEvalRunner, dataset_summary, load_ifeval_examples
from src.evaluator import Evaluator, load_model_configs, load_prompt_configs
from src.reporting import default_analysis_output_dir, default_compare_output_dir, load_run, save_prompt_run_snapshot
from src.utils import PROJECT_ROOT, sanitize_name, utc_now_iso


def _build_evaluate_parser() -> argparse.ArgumentParser:
    """Build the parser for evaluation commands."""
    parser = argparse.ArgumentParser(description="Evaluate local LLMs with Ollama or vLLM.")
    parser.add_argument("--benchmark", choices=["prompts", "ifeval"], default="prompts")
    parser.add_argument("--backend", choices=["ollama", "vllm"], default="ollama")
    parser.add_argument("--models-config", type=Path, default=PROJECT_ROOT / "configs" / "models.yaml")
    parser.add_argument("--prompts-config", type=Path, default=PROJECT_ROOT / "configs" / "prompts.yaml")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--vllm-url", default="http://localhost:8000")
    parser.add_argument("--models", nargs="+", help="Optional model IDs from configs/models.yaml.")
    parser.add_argument("--run-name", help="Optional run name for benchmark outputs.")
    parser.add_argument("--subset-size", type=int, help="Optional number of examples for smoke testing.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature for IFEval.")
    parser.add_argument("--max-tokens", type=int, default=1536, help="Generation max tokens for IFEval.")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def _build_compare_parser() -> argparse.ArgumentParser:
    """Build the parser for comparison commands."""
    parser = argparse.ArgumentParser(description="Compare completed local-llm-eval runs.")
    parser.add_argument("--run-a", type=Path, help="Baseline run path.")
    parser.add_argument("--run-b", type=Path, help="Candidate run path.")
    parser.add_argument("--runs", nargs="+", type=Path, help="Two or more run paths; the first is baseline.")
    parser.add_argument("--out", type=Path, help="Output directory for comparison artifacts.")
    parser.add_argument("--verbose", action="store_true")
    return parser


def _build_analyze_parser() -> argparse.ArgumentParser:
    """Build the parser for analysis commands."""
    parser = argparse.ArgumentParser(description="Analyze a completed local-llm-eval run.")
    parser.add_argument("--run", type=Path, required=True, help="Run path to analyze.")
    parser.add_argument("--out", type=Path, help="Output directory for analysis artifacts.")
    parser.add_argument("--verbose", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Build the CLI parser."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "compare":
        args = _build_compare_parser().parse_args(argv[1:])
        args.command = "compare"
        return args
    if argv and argv[0] == "analyze":
        args = _build_analyze_parser().parse_args(argv[1:])
        args.command = "analyze"
        return args
    args = _build_evaluate_parser().parse_args(argv)
    args.command = "evaluate"
    return args


def configure_logging(verbose: bool) -> None:
    """Initialize application logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    if not verbose:
        for logger_name in ("httpx", "httpcore", "fsspec", "huggingface_hub"):
            logging.getLogger(logger_name).setLevel(logging.WARNING)


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

    if args.command == "compare":
        run_paths = [path for path in (args.runs or [])]
        if args.run_a:
            run_paths.insert(0, args.run_a)
        if args.run_b:
            run_paths.append(args.run_b)
        if len(run_paths) < 2:
            raise RuntimeError("Compare requires --run-a/--run-b or --runs with at least two run paths.")
        output_dir = args.out or default_compare_output_dir(PROJECT_ROOT / "outputs")
        compare_dir = compare_runs(run_paths, output_dir)
        print(f"Saved comparison artifacts to {compare_dir}")
        return 0

    if args.command == "analyze":
        run = load_run(args.run)
        output_dir = args.out or default_analysis_output_dir(PROJECT_ROOT / "outputs", run)
        analysis_dir = analyze_run(args.run, output_dir)
        print(f"Saved analysis artifacts to {analysis_dir}")
        return 0

    summary = build_environment_summary(ollama_url=args.ollama_url, vllm_url=args.vllm_url)
    print_environment_summary(summary)

    if args.check_only:
        return 0

    validate_backend_availability(summary, args.backend)
    models = load_model_configs(
        args.models_config,
        backend_filter=args.backend,
        selected_ids=args.models,
    )

    if args.benchmark == "prompts":
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
        if args.run_name:
            config_snapshot = {
                "benchmark": args.benchmark,
                "backend": args.backend,
                "models_config": str(args.models_config),
                "prompts_config": str(args.prompts_config),
                "selected_models": [model.id for model in models],
                "backend_urls": {
                    "ollama": args.ollama_url,
                    "vllm": args.vllm_url,
                },
                "model_snapshot": [asdict(model) for model in models],
            }
            run_dir = save_prompt_run_snapshot(
                output_root=args.output_dir,
                run_name=args.run_name,
                records=records,
                environment=summary,
                config_snapshot=config_snapshot,
            )
            print(f"Saved prompt-suite run snapshot to {run_dir}")
        return 0

    examples = load_ifeval_examples(limit=args.subset_size)
    benchmark_runner = IFEvalRunner(args.output_dir / "ifeval")
    response_records, result_records = benchmark_runner.run(
        models=models,
        examples=examples,
        backend_urls={"ollama": args.ollama_url, "vllm": args.vllm_url},
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    run_name = args.run_name or f"ifeval-{args.backend}-{utc_now_iso().replace(':', '-')}"
    config_snapshot = {
        "benchmark": args.benchmark,
        "backend": args.backend,
        "models_config": str(args.models_config),
        "selected_models": [model.id for model in models],
        "dataset": {
            "name": "google/IFEval",
            "split": "train",
            "subset_size": args.subset_size,
        },
        "generation": {
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        },
        "backend_urls": {
            "ollama": args.ollama_url,
            "vllm": args.vllm_url,
        },
        "model_snapshot": [asdict(model) for model in models],
    }
    run_dir = benchmark_runner.save(
        run_name=sanitize_name(run_name),
        response_records=response_records,
        result_records=result_records,
        config_snapshot=config_snapshot,
        environment=summary,
        dataset_info=dataset_summary(examples),
    )
    print(f"Saved IFEval run with {len(result_records)} records to {run_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
