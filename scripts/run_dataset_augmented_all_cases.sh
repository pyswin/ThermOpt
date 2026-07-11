#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CASE_ROOT="${ROOT_DIR}/external/ATPlace_pub/cases"
OUTPUT_ROOT="${ROOT_DIR}/outputs/thermopt_dataset_augmented_500"

CASES=(1 2 3 4 5 6 7 8 9 10)
NUM_SAMPLES="${NUM_SAMPLES:-500}"
LAYOUT_MODE="${LAYOUT_MODE:-random}"
SAVE_FORMATS="${SAVE_FORMATS:-pointwise_augmented,json}"
MIN_GAP="${MIN_GAP:-0.05}"
UNIT_SCALE="${UNIT_SCALE:-0.001}"
INITIAL_LAYOUT="${INITIAL_LAYOUT:-pl}"
SEED_BASE="${SEED_BASE:-7}"

RANDOMIZE_POSITION="${RANDOMIZE_POSITION:-true}"
RANDOMIZE_POWER="${RANDOMIZE_POWER:-true}"
RANDOMIZE_ROTATION="${RANDOMIZE_ROTATION:-true}"

mkdir -p "${OUTPUT_ROOT}"

for case_id in "${CASES[@]}"; do
  case_dir="${CASE_ROOT}/Case${case_id}"
  out_dir="${OUTPUT_ROOT}/case${case_id}_augmented"
  seed=$((SEED_BASE + case_id * 1000))

  rm -rf "${out_dir}"
  mkdir -p "${out_dir}"

  cmd=(
    python3 "${ROOT_DIR}/scripts/generate_thermal_dataset.py"
    --case_dir "${case_dir}"
    --output_dir "${out_dir}"
    --num_samples "${NUM_SAMPLES}"
    --variation_type random
    --layout_mode "${LAYOUT_MODE}"
    --save_formats "${SAVE_FORMATS}"
    --min_gap "${MIN_GAP}"
    --unit_scale "${UNIT_SCALE}"
    --initial_layout "${INITIAL_LAYOUT}"
    --seed "${seed}"
  )

  if [[ "${RANDOMIZE_POSITION}" == "true" ]]; then
    cmd+=(--randomize_position)
  else
    cmd+=(--no-randomize_position)
  fi

  if [[ "${RANDOMIZE_POWER}" == "true" ]]; then
    cmd+=(--randomize_power)
  else
    cmd+=(--no-randomize_power)
  fi

  if [[ "${RANDOMIZE_ROTATION}" == "true" ]]; then
    cmd+=(--randomize_rotation)
  else
    cmd+=(--no-randomize_rotation)
  fi

  echo "[info] generating Case${case_id} -> ${out_dir}"
  "${cmd[@]}"
done

echo "[done] augmented datasets are in: ${OUTPUT_ROOT}"
