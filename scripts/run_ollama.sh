#!/usr/bin/env bash
set -euo pipefail

python -m src.runner --backend ollama "$@"

