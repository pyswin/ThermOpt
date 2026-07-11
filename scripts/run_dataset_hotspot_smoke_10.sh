#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CASE_ROOT="${ROOT_DIR}/external/ATPlace_pub/cases"
OUTPUT_ROOT="${ROOT_DIR}/outputs/thermopt_hotspot_smoke_10"

CASES=(1 2 3 4 5 6 7 8 9 10)
NUM_SAMPLES="${NUM_SAMPLES:-10}"
LAYOUT_MODE="${LAYOUT_MODE:-random}"
SAVE_FORMATS="${SAVE_FORMATS:-pointwise_augmented,json}"
MIN_GAP="${MIN_GAP:-0.05}"
UNIT_SCALE="${UNIT_SCALE:-0.001}"
INITIAL_LAYOUT="${INITIAL_LAYOUT:-pl}"
SEED_BASE="${SEED_BASE:-7}"

RANDOMIZE_POSITION="${RANDOMIZE_POSITION:-true}"
RANDOMIZE_POWER="${RANDOMIZE_POWER:-true}"
RANDOMIZE_ROTATION="${RANDOMIZE_ROTATION:-true}"
BACKEND="${BACKEND:-hotspot}"
HOTSPOT_REQUIRED="${HOTSPOT_REQUIRED:-true}"
HOTSPOT_ALLOW_FALLBACK="${HOTSPOT_ALLOW_FALLBACK:-false}"

rm -rf "${OUTPUT_ROOT}"
mkdir -p "${OUTPUT_ROOT}"

for case_id in "${CASES[@]}"; do
  case_dir="${CASE_ROOT}/Case${case_id}"
  out_dir="${OUTPUT_ROOT}/case${case_id}"
  seed=$((SEED_BASE + case_id * 1000))

  echo "[info] generating Case${case_id} -> ${out_dir}"
  python3 "${ROOT_DIR}/scripts/generate_thermal_dataset.py" \
    --case_dir "${case_dir}" \
    --output_dir "${out_dir}" \
    --num_samples "${NUM_SAMPLES}" \
    --variation_type random \
    --layout_mode "${LAYOUT_MODE}" \
    --save_formats "${SAVE_FORMATS}" \
    --min_gap "${MIN_GAP}" \
    --unit_scale "${UNIT_SCALE}" \
    --initial_layout "${INITIAL_LAYOUT}" \
    --seed "${seed}" \
    --backend "${BACKEND}" \
    --hotspot_binary "${ROOT_DIR}/external/ATPlace_pub/thermal/hotspot" \
    $( [[ "${HOTSPOT_REQUIRED}" == "true" ]] && printf '%s ' --hotspot_required || printf '%s ' --no-hotspot_required ) \
    $( [[ "${HOTSPOT_ALLOW_FALLBACK}" == "true" ]] && printf '%s ' --hotspot_allow_fallback || printf '%s ' --no-hotspot_allow_fallback ) \
    $( [[ "${RANDOMIZE_POSITION}" == "true" ]] && printf '%s ' --randomize_position || printf '%s ' --no-randomize_position ) \
    $( [[ "${RANDOMIZE_POWER}" == "true" ]] && printf '%s ' --randomize_power || printf '%s ' --no-randomize_power ) \
    $( [[ "${RANDOMIZE_ROTATION}" == "true" ]] && printf '%s ' --randomize_rotation || printf '%s ' --no-randomize_rotation )

  python3 - <<PY
from pathlib import Path
import json

out_dir = Path("${out_dir}")
summary_path = out_dir / "dataset_summary.json"
if not summary_path.is_file():
    raise SystemExit(f"missing summary: {summary_path}")

summary = json.loads(summary_path.read_text(encoding="utf-8"))
pointwise_dir = out_dir / "pointwise"
json_dir = out_dir / "json"

if summary.get("successful_samples") != ${NUM_SAMPLES}:
    raise SystemExit(f"unexpected successful_samples: {summary.get('successful_samples')}")
    if summary.get("thermal_backend", {}).get("requested") != "hotspot":
        raise SystemExit(f"hotspot backend not recorded in summary: {summary.get('thermal_backend')}")
    augmented_dir = out_dir / "pointwise_augmented"
    if len(list(augmented_dir.glob("sample_*.csv"))) != ${NUM_SAMPLES}:
        raise SystemExit(f"unexpected augmented sample count in {augmented_dir}")
    if len(list(json_dir.glob("sample_*.json"))) != ${NUM_SAMPLES}:
        raise SystemExit(f"unexpected json sample count in {json_dir}")

print(json.dumps({
    "case": summary.get("case"),
    "successful_samples": summary.get("successful_samples"),
    "backend": summary.get("thermal_backend"),
}, indent=2, ensure_ascii=False))
PY
done

echo "[done] smoke test datasets are in: ${OUTPUT_ROOT}"
