#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CASE_ROOT="${ROOT_DIR}/external/ATPlace_pub/cases"
OUTPUT_ROOT="${ROOT_DIR}/outputs/thermopt_dataset_qc500"

CASES=(1 2 3 4 5 6 7 8 9 10)
TARGET_TOTAL="${TARGET_TOTAL:-500}"
MODE_RANDOM="${MODE_RANDOM:-300}"
MODE_DISPERSED="${MODE_DISPERSED:-100}"
MODE_CLUSTER="${MODE_CLUSTER:-50}"
MODE_BOUNDARY="${MODE_BOUNDARY:-50}"
MIN_GAP="${MIN_GAP:-0.05}"
UNIT_SCALE="${UNIT_SCALE:-0.001}"
INITIAL_LAYOUT="${INITIAL_LAYOUT:-pl}"
STAGE_SAVE_FORMATS="${STAGE_SAVE_FORMATS:-pointwise_augmented,json}"
FINAL_EXPORT_JSON="${FINAL_EXPORT_JSON:-false}"
SEED_BASE="${SEED_BASE:-6}"
BATCH_MULTIPLIER="${BATCH_MULTIPLIER:-1.5}"
MAX_BATCHES="${MAX_BATCHES:-20}"

RANDOMIZE_POSITION="${RANDOMIZE_POSITION:-true}"
RANDOMIZE_POWER="false"
RANDOMIZE_ROTATION="${RANDOMIZE_ROTATION:-true}"
BACKEND="${BACKEND:-hotspot}"
HOTSPOT_REQUIRED="${HOTSPOT_REQUIRED:-true}"
HOTSPOT_ALLOW_FALLBACK="${HOTSPOT_ALLOW_FALLBACK:-false}"

rm -rf "${OUTPUT_ROOT}"
mkdir -p "${OUTPUT_ROOT}"

for case_id in "${CASES[@]}"; do
  case_dir="${CASE_ROOT}/Case${case_id}"
  final_case_dir="${OUTPUT_ROOT}/case${case_id}"
  staging_case_dir="${OUTPUT_ROOT}/_staging/case${case_id}"

  rm -rf "${final_case_dir}" "${staging_case_dir}"
  mkdir -p "${final_case_dir}/pointwise_augmented" "${staging_case_dir}"
  if [[ "${FINAL_EXPORT_JSON}" == "true" ]]; then
    mkdir -p "${final_case_dir}/json"
  fi

  echo "[info] generating case${case_id}"
  CASE_ID="${case_id}" \
  CASE_DIR="${case_dir}" \
  ROOT_DIR="${ROOT_DIR}" \
  FINAL_DIR="${final_case_dir}" \
  STAGING_DIR="${staging_case_dir}" \
  TARGET_TOTAL="${TARGET_TOTAL}" \
  MODE_RANDOM="${MODE_RANDOM}" \
  MODE_DISPERSED="${MODE_DISPERSED}" \
  MODE_CLUSTER="${MODE_CLUSTER}" \
  MODE_BOUNDARY="${MODE_BOUNDARY}" \
  MIN_GAP="${MIN_GAP}" \
  UNIT_SCALE="${UNIT_SCALE}" \
  INITIAL_LAYOUT="${INITIAL_LAYOUT}" \
  STAGE_SAVE_FORMATS="${STAGE_SAVE_FORMATS}" \
  FINAL_EXPORT_JSON="${FINAL_EXPORT_JSON}" \
  SEED_BASE="${SEED_BASE}" \
  BATCH_MULTIPLIER="${BATCH_MULTIPLIER}" \
  MAX_BATCHES="${MAX_BATCHES}" \
  RANDOMIZE_POSITION="${RANDOMIZE_POSITION}" \
  RANDOMIZE_POWER="${RANDOMIZE_POWER}" \
  RANDOMIZE_ROTATION="${RANDOMIZE_ROTATION}" \
  BACKEND="${BACKEND}" \
  HOTSPOT_REQUIRED="${HOTSPOT_REQUIRED}" \
  HOTSPOT_ALLOW_FALLBACK="${HOTSPOT_ALLOW_FALLBACK}" \
  python3 - <<'PY'
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

case_id = int(os.environ["CASE_ID"])
case_dir = Path(os.environ["CASE_DIR"])
root_dir = Path(os.environ["ROOT_DIR"])
final_dir = Path(os.environ["FINAL_DIR"])
staging_dir = Path(os.environ["STAGING_DIR"])

target_total = int(os.environ["TARGET_TOTAL"])
mode_targets = {
    "random": int(os.environ["MODE_RANDOM"]),
    "dispersed": int(os.environ["MODE_DISPERSED"]),
    "cluster": int(os.environ["MODE_CLUSTER"]),
    "boundary": int(os.environ["MODE_BOUNDARY"]),
}
min_gap = float(os.environ["MIN_GAP"])
unit_scale = float(os.environ["UNIT_SCALE"])
initial_layout = os.environ["INITIAL_LAYOUT"]
stage_save_formats = os.environ["STAGE_SAVE_FORMATS"]
final_export_json = os.environ["FINAL_EXPORT_JSON"].lower() == "true"
seed_base = int(os.environ["SEED_BASE"])
batch_multiplier = float(os.environ["BATCH_MULTIPLIER"])
max_batches = int(os.environ["MAX_BATCHES"])
randomize_position = os.environ["RANDOMIZE_POSITION"].lower() == "true"
randomize_power = os.environ["RANDOMIZE_POWER"].lower() == "true"
randomize_rotation = os.environ["RANDOMIZE_ROTATION"].lower() == "true"
backend = os.environ["BACKEND"]
hotspot_required = os.environ["HOTSPOT_REQUIRED"].lower() == "true"
hotspot_allow_fallback = os.environ["HOTSPOT_ALLOW_FALLBACK"].lower() == "true"

generator_script = root_dir / "scripts" / "dataset_gen" / "generate_thermal_dataset.py"
python = sys.executable or "python3"

mode_seed_offsets = {
    "random": 0,
    "dispersed": 10000,
    "cluster": 20000,
    "boundary": 30000,
}

def build_common_args(out_dir: Path, mode: str, num_samples: int, seed: int) -> list[str]:
    args = [
        python,
        str(generator_script),
        "--case_dir",
        str(case_dir),
        "--output_dir",
        str(out_dir),
        "--num_samples",
        str(num_samples),
        "--variation_type",
        "random",
        "--layout_mode",
        mode,
        "--save_formats",
        stage_save_formats,
        "--min_gap",
        str(min_gap),
        "--unit_scale",
        str(unit_scale),
        "--initial_layout",
        initial_layout,
        "--seed",
        str(seed),
        "--backend",
        backend,
        "--hotspot_binary",
        str(root_dir / "external" / "ATPlace_pub" / "thermal" / "hotspot"),
    ]
    if hotspot_required:
        args.append("--hotspot_required")
    else:
        args.append("--no-hotspot_required")
    if hotspot_allow_fallback:
        args.append("--hotspot_allow_fallback")
    else:
        args.append("--no-hotspot_allow_fallback")
    if randomize_position:
        args.append("--randomize_position")
    else:
        args.append("--no-randomize_position")
    if randomize_power:
        args.append("--randomize_power")
    else:
        args.append("--no-randomize_power")
    if randomize_rotation:
        args.append("--randomize_rotation")
    else:
        args.append("--no-randomize_rotation")
    return args

def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def chiplet_signature(cfg: dict) -> tuple:
    cx = float(cfg.get("cx", float(cfg["x"]) + float(cfg["width"]) * 0.5))
    cy = float(cfg.get("cy", float(cfg["y"]) + float(cfg["height"]) * 0.5))
    return (
        str(cfg.get("name", "")),
        round(cx, 2),
        round(cy, 2),
        round(float(cfg["width"]), 2),
        round(float(cfg["height"]), 2),
        int(cfg.get("rotation", 0)) % 360,
    )

def layout_fingerprint(sample: dict) -> str:
    configs = sample["chiplet_configs"]
    payload = sorted(chiplet_signature(cfg) for cfg in configs)
    digest = hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()
    return digest

def _group_rows(centers: list[tuple[float, float]], row_tol: float) -> list[list[tuple[float, float]]]:
    rows: list[list[tuple[float, float]]] = []
    for cx, cy in sorted(centers, key=lambda item: (item[1], item[0])):
        if not rows:
            rows.append([(cx, cy)])
            continue
        if abs(cy - rows[-1][-1][1]) <= row_tol:
            rows[-1].append((cx, cy))
        else:
            rows.append([(cx, cy)])
    return rows

def is_shelf_like(sample: dict) -> bool:
    configs = sample["chiplet_configs"]
    n = len(configs)
    if n < 8:
        return False

    centers = []
    for cfg in configs:
        cx = float(cfg.get("cx", float(cfg["x"]) + float(cfg["width"]) * 0.5))
        cy = float(cfg.get("cy", float(cfg["y"]) + float(cfg["height"]) * 0.5))
        centers.append((cx, cy))

    row_tol = max(0.05, min_gap * 0.8)
    rows = _group_rows(centers, row_tol)
    if len(rows) > 4:
        return False

    largest_row = max(len(row) for row in rows)
    if largest_row < max(6, n // 3):
        return False

    row_centers = [sum(item[1] for item in row) / len(row) for row in rows]
    if len(row_centers) < 2:
        return True

    row_centers = sorted(row_centers)
    gaps = [row_centers[i + 1] - row_centers[i] for i in range(len(row_centers) - 1)]
    mean_gap = sum(gaps) / len(gaps)
    if mean_gap <= 0.0:
        return True
    variance = sum((gap - mean_gap) ** 2 for gap in gaps) / len(gaps)
    std_gap = math.sqrt(variance)
    return std_gap <= max(0.02, mean_gap * 0.35)

def copy_sample(batch_dir: Path, sample_stem: str, dst_index: int) -> None:
    src_csv = batch_dir / "pointwise_augmented" / f"{sample_stem}.csv"
    dst_csv = final_dir / "pointwise_augmented" / f"sample_{dst_index:06d}.csv"
    shutil.copy2(src_csv, dst_csv)
    if final_export_json:
        src_json = batch_dir / "json" / f"{sample_stem}.json"
        dst_json = final_dir / "json" / f"sample_{dst_index:06d}.json"
        shutil.copy2(src_json, dst_json)

staging_case_dir = staging_dir
seen_fingerprints: set[str] = set()
accepted_counts = {mode: 0 for mode in mode_targets}
rejected_duplicate = 0
rejected_shelf = 0
generated_batches = 0
global_index = 0

for mode in ("random", "dispersed", "cluster", "boundary"):
    target = mode_targets[mode]
    accepted = 0
    batch_index = 0
    while accepted < target and batch_index < max_batches:
        remaining = target - accepted
        batch_size = max(50, int(math.ceil(remaining * batch_multiplier)))
        seed = seed_base + case_id * 100000 + mode_seed_offsets[mode] + batch_index
        batch_dir = staging_case_dir / f"{mode}_{batch_index:03d}"
        batch_dir.mkdir(parents=True, exist_ok=True)

        cmd = build_common_args(batch_dir, mode, batch_size, seed)
        subprocess.run(cmd, check=True)
        generated_batches += 1

        json_dir = batch_dir / "json"
        for sample_json in sorted(json_dir.glob("sample_*.json")):
            if accepted >= target:
                break
            sample = load_json(sample_json)
            fingerprint = layout_fingerprint(sample)
            if fingerprint in seen_fingerprints:
                rejected_duplicate += 1
                continue
            if is_shelf_like(sample):
                rejected_shelf += 1
                continue

            copy_sample(batch_dir, sample_json.stem, global_index)
            seen_fingerprints.add(fingerprint)
            accepted += 1
            accepted_counts[mode] += 1
            global_index += 1

        shutil.rmtree(batch_dir, ignore_errors=True)
        batch_index += 1

    if accepted < target:
        raise SystemExit(
            f"case{case_id} mode={mode} only accepted {accepted}/{target} samples after {batch_index} batches"
        )

summary = {
    "case": case_dir.name,
    "case_id": case_id,
    "requested_samples": target_total,
    "successful_samples": global_index,
    "mode_targets": mode_targets,
    "mode_accepted": accepted_counts,
    "rejected_duplicate": rejected_duplicate,
    "rejected_shelf_like": rejected_shelf,
    "generated_batches": generated_batches,
    "layout_quality_rules": {
        "duplicate_fingerprint_round_mm": 0.01,
        "shelf_row_tolerance_mm": max(0.05, min_gap * 0.8),
    },
    "generation_config": {
        "min_gap": min_gap,
        "unit_scale": unit_scale,
        "initial_layout": initial_layout,
        "stage_save_formats": stage_save_formats,
        "final_export_json": final_export_json,
        "backend": backend,
        "randomize_position": randomize_position,
        "randomize_power": randomize_power,
        "randomize_rotation": randomize_rotation,
        "seed_base": seed_base,
        "batch_multiplier": batch_multiplier,
        "max_batches": max_batches,
    },
}
(final_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY
done

rm -rf "${OUTPUT_ROOT}/_staging"
echo "[done] filtered augmented datasets are in: ${OUTPUT_ROOT}"
