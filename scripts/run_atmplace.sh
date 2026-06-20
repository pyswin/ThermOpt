#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export MPLCONFIGDIR="$ROOT_DIR/.cache/matplotlib"
export XDG_CACHE_HOME="$ROOT_DIR/.cache"
mkdir -p "$MPLCONFIGDIR"

PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

PYTHONPATH=src "$PYTHON_BIN" -m thermopt.experiments.run_optimizer_comparison --config configs/atmplace_benchmark.yaml
