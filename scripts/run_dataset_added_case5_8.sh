#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CASE_ROOT="${ROOT_DIR}/external/ATPlace_pub/cases"
OUTPUT_ROOT="${ROOT_DIR}/outputs/dataset_added"
ZIP_PATH="${OUTPUT_ROOT}/augmented_datasets.zip"

mkdir -p "${OUTPUT_ROOT}"

rm -rf \
  "${OUTPUT_ROOT}/case5" \
  "${OUTPUT_ROOT}/case6" \
  "${OUTPUT_ROOT}/case7" \
  "${OUTPUT_ROOT}/case8" \
  "${OUTPUT_ROOT}/case5_augmented" \
  "${OUTPUT_ROOT}/case6_augmented" \
  "${OUTPUT_ROOT}/case7_augmented" \
  "${OUTPUT_ROOT}/case8_augmented" \
  "${ZIP_PATH}"

for case_id in 5 6 7 8; do
  echo "Generating Case${case_id}"
  python3 "${ROOT_DIR}/scripts/generate_thermal_dataset.py" \
    --case_dir "${CASE_ROOT}/Case${case_id}" \
    --output_dir "${OUTPUT_ROOT}/case${case_id}" \
    --num_samples 1000 \
    --variation_type random \
    --no-randomize_power
done

for case_id in 5 6 7 8; do
  echo "Augmenting case${case_id}"
  python3 "${ROOT_DIR}/scripts/augment_pointwise_features.py" \
    --dataset_dir "${OUTPUT_ROOT}/case${case_id}" \
    --output_dir "${OUTPUT_ROOT}/case${case_id}_augmented"
done

python3 - <<'PY'
from pathlib import Path
import zipfile

root = Path("/home/ywk/ThermOpt/outputs/dataset_added")
zip_path = root / "augmented_datasets.zip"
case_dirs = [root / f"case{n}_augmented" for n in (5, 6, 7, 8)]

for case_dir in case_dirs:
    if not case_dir.is_dir():
        raise SystemExit(f"Missing augmented dir: {case_dir}")

if zip_path.exists():
    zip_path.unlink()

with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for case_dir in case_dirs:
        for file_path in case_dir.rglob("*"):
            if file_path.is_file():
                zf.write(file_path, arcname=file_path.relative_to(root))

print(zip_path)
PY

rm -rf \
  "${OUTPUT_ROOT}/case5" \
  "${OUTPUT_ROOT}/case6" \
  "${OUTPUT_ROOT}/case7" \
  "${OUTPUT_ROOT}/case8" \
  "${OUTPUT_ROOT}/case5_augmented" \
  "${OUTPUT_ROOT}/case6_augmented" \
  "${OUTPUT_ROOT}/case7_augmented" \
  "${OUTPUT_ROOT}/case8_augmented"

echo "Done. Final archive: ${ZIP_PATH}"
