"""
hotspot_dataset_check.py — sanity-check that milp_wl_dataset records
(scripts/run_milp_dataset.py) can be fed into the HotSpot thermal backend.

The dataset JSON itself isn't a type HotSpotBackend understands -- this uses
thermopt.data.dataset_io.layout_from_chiplets_json to convert a record's
"chiplets" list back into a Layout first, then runs the same HotSpotBackend
used elsewhere in the repo (thermopt.thermal.hotspot) unchanged. No dataset
generation happens here, this only proves the round-trip works.

Usage:
    python scripts/hotspot_dataset_check.py <run_dir> <CaseN> [--index N] [--source perturb|intermediate]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from thermopt.data.atplace import load_atplace_case
from thermopt.data.dataset_io import layout_from_chiplets_json
from thermopt.layout.geometry import total_outline_penalty, total_overlap_penalty
from thermopt.layout.visualization import save_temperature_figure
from thermopt.thermal.hotspot import HotSpotBackend

CASES_DIR = ROOT / "external/ATPlace_pub/cases"
HS_BIN = ROOT / "external/ATPlace_pub/thermal/hotspot"


def main(run_dir: Path, case_name: str, source: str, index: int | None) -> None:
    ci = load_atplace_case(CASES_DIR / case_name, {}, seed=42)
    case = ci.case

    if source == "perturb":
        data_path = run_dir / "perturb" / case_name / "perturbations.json"
        records = json.loads(data_path.read_text())
        legal = [r for r in records if r["legal"]]
        record = legal[index] if index is not None else legal[0]
        label = f"perturb #{record['id']} ({record['move_type']})"
    else:
        data_path = run_dir / "intermediate" / case_name / "snapshots.json"
        records = json.loads(data_path.read_text())
        record = records[index] if index is not None else records[-1]
        label = f"snapshot t={record['t']:.1f}s"

    layout = layout_from_chiplets_json(record["chiplets"])
    overlap = total_overlap_penalty(case, layout)
    outline = total_outline_penalty(case, layout)
    print(f"{case_name} {label}: converted from JSON OK, overlap={overlap:.6f} outline={outline:.6f}")

    backend = HotSpotBackend(case=case, config={"hotspot_binary": str(HS_BIN)})
    t0 = time.time()
    grid = backend.simulate(case, layout)
    elapsed = time.time() - t0
    flat_sorted = sorted(grid.flatten(), reverse=True)
    tmax = flat_sorted[0]
    top5 = sum(flat_sorted[: max(1, len(flat_sorted) // 20)]) / max(1, len(flat_sorted) // 20)
    print(f"HotSpot simulate() OK in {elapsed:.1f}s  grid={grid.shape}  Tmax={tmax:.2f}C  top5%={top5:.2f}C  Tmin={min(flat_sorted):.2f}C")

    out_path = run_dir / "perturb" / case_name / "hotspot_check.png" if source == "perturb" else run_dir / "intermediate" / case_name / "hotspot_check.png"
    save_temperature_figure(grid, out_path, f"{case_name} {label}  HotSpot Tmax={tmax:.1f}C")
    print("saved", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("case", type=str)
    parser.add_argument("--source", choices=["perturb", "intermediate"], default="perturb")
    parser.add_argument("--index", type=int, default=None)
    args = parser.parse_args()
    main(args.run_dir, args.case, args.source, args.index)
