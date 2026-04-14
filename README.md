# local-llm-eval

`local-llm-eval` is a small Python evaluation harness for comparing local LLMs with Ollama as the default backend and vLLM as an optional backend.

It runs a small YAML-defined prompt suite across local models and saves structured results to `outputs/results.csv` and `outputs/results.json`.

## What It Measures

- Latency
- Tokens per second
- Response length
- Simple rubric score
- Short-context vs longer-context behavior
- Low vs higher temperature behavior

## Project Layout

```text
local-llm-eval/
├── configs/
│   ├── models.yaml
│   └── prompts.yaml
├── outputs/
├── scripts/
│   ├── run_ollama.sh
│   └── run_vllm.sh
├── src/
│   ├── backends/
│   │   ├── ollama_backend.py
│   │   └── vllm_backend.py
│   ├── env_check.py
│   ├── evaluator.py
│   ├── rubric.py
│   ├── runner.py
│   └── utils.py
├── environment.yml
├── requirements.txt
└── .gitignore
```

## Dependency Notes

This project was built to reuse the existing Conda environment named `local-llm-eval`.

Minimal baseline Python dependencies:

- `PyYAML`
- `requests`
- `psutil`

Optional dependencies:

- `torch` for richer CUDA detection
- `vllm` if you want to use the optional vLLM path

## Setup

If your Conda environment already exists, reuse it:

```bash
conda activate local-llm-eval
python --version
```

If you want to align the environment to the project file:

```bash
conda env update -n local-llm-eval -f environment.yml
```

## Verify The Environment

Run:

```bash
python -m src.runner --check-only
```

This prints a clean summary including:

- OS
- Python version
- CPU and RAM
- `torch` CUDA availability if installed
- GPU details from `torch` and `nvidia-smi` when available
- Whether `ollama` is installed and reachable
- Whether `vllm` is installed

## Recommended Models For 24GB VRAM

These are preloaded in `configs/models.yaml` for Ollama:

- `qwen2.5:7b`
- `llama3.1:8b`
- `gemma2:9b`

Optional disabled vLLM examples are also included in the config.

## Ollama Mode

Ollama is the default path.

Start Ollama separately, make sure the models exist locally, then run:

```bash
python -m src.runner --backend ollama
```

Or with the helper script:

```bash
bash scripts/run_ollama.sh
```

Example with a custom Ollama URL:

```bash
python -m src.runner --backend ollama --ollama-url http://localhost:11434
```

## vLLM Mode

vLLM is optional and intentionally minimal.

This project assumes a vLLM OpenAI-compatible server is already running. The backend stays disabled unless you explicitly request it.

Run:

```bash
python -m src.runner --backend vllm --vllm-url http://localhost:8000
```

Or with the helper script:

```bash
bash scripts/run_vllm.sh
```

## Example Commands

```bash
python -m src.runner --check-only
python -m src.runner --backend ollama
python -m src.runner --backend ollama --models-config configs/models.yaml --prompts-config configs/prompts.yaml
python -m src.runner --backend vllm --vllm-url http://localhost:8000
```

## Example Outputs

CSV columns include:

- `model_id`
- `backend`
- `prompt_id`
- `temperature`
- `latency_seconds`
- `tokens_per_second`
- `response_length_chars`
- `response_length_words`
- `rubric_score`
- `error`

JSON output stores the same records plus a summary block and environment metadata.

## Notes

- Ollama is the default and recommended local path.
- vLLM is optional and isolated behind its own backend file.
- If `torch` or `vllm` are not installed, the project falls back cleanly instead of failing during environment checks.

