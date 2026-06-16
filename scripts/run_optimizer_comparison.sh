#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export MPLCONFIGDIR="$ROOT_DIR/.cache/matplotlib"
export XDG_CACHE_HOME="$ROOT_DIR/.cache"
mkdir -p "$MPLCONFIGDIR"

PYTHONPATH=src python -m thermopt.experiments.run_optimizer_comparison --config configs/optimizer_comparison.yaml
