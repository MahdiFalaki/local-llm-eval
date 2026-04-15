# local-llm-eval

`local-llm-eval` is a lightweight local evaluation pipeline for served LLMs. It is built for practical offline evaluation with Ollama first, optional vLLM support, YAML-defined prompt suites, benchmark-style IFEval runs, and single-run analysis artifacts that are easy to inspect and reuse in notes or a README.

## What It Supports

- Prompt-suite evaluation from YAML prompt and model configs
- IFEval benchmark evaluation using the `google/IFEval` dataset
- Single-run analysis from an already completed run
- Offline artifacts in JSON, CSV, JSONL, and Markdown

## Main Workflow

1. Check the local environment
2. Run an evaluation
3. Analyze the completed run

## Project Structure

- `configs/` - model and prompt configuration
- `src/runner.py` - CLI entrypoint
- `src/backends/` - Ollama and optional vLLM backends
- `src/ifeval.py` - IFEval dataset loading and benchmark execution
- `src/analyze.py` - single-run analysis and reporting
- `outputs/` - saved runs and analysis artifacts

## Setup

Reuse the existing Conda environment if it is already available:

```bash
conda activate local-llm-eval
python --version
```

If you want to align the environment with the project file:

```bash
conda env update -n local-llm-eval -f environment.yml
```

## Environment Check

```bash
python -m src.runner --check-only
```

## Recommended Models

- `qwen2.5:7b`
- `llama3.1:8b`
- `gemma2:9b`

## How to Run

### Prompt-Suite Evaluation

```bash
python -m src.runner \
  --backend ollama \
  --models qwen2_5_7b llama3_1_8b gemma2_9b \
  --run-name prompts-full
```

### IFEval Smoke Run

```bash
python -m src.runner \
  --benchmark ifeval \
  --backend ollama \
  --models qwen2_5_7b \
  --subset-size 3 \
  --run-name ifeval-smoke
```

### IFEval Full Run

```bash
python -m src.runner \
  --benchmark ifeval \
  --backend ollama \
  --models qwen2_5_7b llama3_1_8b gemma2_9b \
  --run-name ifeval-full
```

### Analyze Completed Run

```bash
python -m src.runner analyze \
  --run outputs/ifeval/ifeval-full \
  --out outputs/analysis/ifeval-full
```

## Analysis Outputs

The single-run analysis step writes Markdown and CSV artifacts that are directly usable in project notes or a README:

- `model_summary.md` / `model_summary.csv` - top-level model overview
- `failure_breakdown.md` / `failure_breakdown.csv` - where each model fails
- `analysis_report.md` - short human-readable summary

These artifacts matter because they cover three separate jobs cleanly:

- overview: which model performed best overall
- diagnosis: where each model tends to fail
- summary: a compact write-up that is easy to reuse

## Example Results

Example single-run IFEval summary from the generated analysis artifacts:

| Model | Backend | Strict | Loose | Avg Latency | Avg TPS | Best Category | Worst Category |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `gemma2_9b` | `ollama` | 0.7283 | 0.7579 | 2.6511 | 116.2692 | `language` | `length_constraints` |
| `llama3_1_8b` | `ollama` | 0.7264 | 0.7652 | 2.4694 | 145.1115 | `language` | `combination` |
| `qwen2_5_7b` | `ollama` | 0.7079 | 0.7338 | 2.0939 | 152.6595 | `language` | `length_constraints` |

This is the role of `model_summary.md`: a compact overview of quality, speed, and category-level strengths and weaknesses for one completed run.

## Backend Notes

- Ollama is the default and primary backend
- vLLM is optional and only needed if you want the alternate serving path

## Minimal Recommended Workflow

1. Run full IFEval once
2. Analyze the completed run
3. Reuse the generated Markdown and CSV artifacts in your README or notes

## Short Notes

- Offline and file-based
- Small and reproducible
- No judge-model dependency
- Ollama-first
