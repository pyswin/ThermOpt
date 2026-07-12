#!/usr/bin/env python3
"""Convert MILP-run layout JSONs (atplace/milp_runs/<run>/CaseN/summary.json) into
the "pointwise_augmented" CSV format used elsewhere in the dataset pipeline
(grid_x, grid_y, chiplet_id, chiplet_power, temperature, occupancy_mask,
edge_mask, coord_x_norm, coord_y_norm).

By default this re-simulates the layout with HotSpot to fill in real
temperatures. With --skip_hotspot it skips the simulation entirely and writes
0.0 as a placeholder in the temperature column -- useful when you only need
the layout/power input features for model inference and don't want to pay for
a HotSpot run.

Unless --skip_hotspot is set, this also saves the raw (rows, cols) HotSpot
temperature grid per case (hotspot_grid.npy, Celsius) and a thermal_summary.json
with Tmax/T50/Tmean/Tstd -- so RMSE/T50 against a model prediction, or a
visualization, can be computed downstream without re-running HotSpot.

Usage:
  python3 scripts/dataset_gen/json_to_augmented_csv.py \
      --milp_dir atplace/milp_runs/20260704_025209_gurobi_t150s_hotspot \
      --out_dir atplace/hotspot_augmented_dataset/from_milp_runs

  python3 scripts/dataset_gen/json_to_augmented_csv.py \
      --milp_dir atplace/milp_runs/20260704_025209_gurobi_t150s_hotspot \
      --out_dir atplace/hotspot_augmented_dataset/from_milp_runs_no_temp \
      --skip_hotspot
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from thermopt.data.dataset_io import layout_from_chiplets_json
from thermopt.data.thermal_dataset import ThermalDatasetGenerator, ThermalSample

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CASES_DIR = REPO_ROOT / "external/ATPlace_pub/cases"
TOP_K = 50


def top_k_mean(grid: np.ndarray, k: int = TOP_K) -> float:
    """Mean of the k hottest grid cells (T50 when k=50)."""
    flat = grid.flatten()
    k = min(k, flat.size)
    return float(np.mean(np.sort(flat)[-k:]))


def discover_cases(milp_dir: Path) -> list[str]:
    return sorted(
        (p.name for p in milp_dir.iterdir() if p.is_dir() and (p / "summary.json").exists()),
        key=lambda name: int("".join(filter(str.isdigit, name)) or 0),
    )


def build_sample(generator: ThermalDatasetGenerator, chiplets: list[dict], skip_hotspot: bool) -> ThermalSample | None:
    layout = layout_from_chiplets_json(chiplets)
    configs = ThermalDatasetGenerator._layout_to_configs(generator.case, layout)

    if skip_hotspot:
        rows, cols = generator.thermal_config["grid_size"]
        temp_map = np.zeros((rows, cols), dtype=np.float32)
    else:
        # These layouts are already-solved MILP output (legal=True, overlap_penalty=0.0 in
        # their own summary.json), so we simulate directly instead of going through
        # generate_sample()'s _check_chiplet_collision -- that gate is meant to validate
        # freshly-generated candidate layouts and has no floating-point tolerance, so it can
        # reject an already-legal touching (zero-gap) layout on fp noise alone.
        try:
            temp_map = generator.backend.simulate(generator.case, layout)
        except Exception as exc:
            print(f"[error] HotSpot simulation failed: {exc}")
            return None

    return ThermalSample(
        sample_id=0,
        timestamp=datetime.now().isoformat(),
        variation_type="milp_run",
        layout_mode="fixed",
        chiplet_configs=configs,
        temperature_map=temp_map,
        temperature_stats={
            "min": float(temp_map.min()),
            "max": float(temp_map.max()),
            "mean": float(temp_map.mean()),
            "std": float(temp_map.std()),
        },
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--milp_dir", type=Path, required=True, help="Directory with CaseN/summary.json layouts")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--cases", nargs="+", default=None, help="Subset of case names, e.g. Case1 Case2 (default: auto-discover)")
    ap.add_argument("--skip_hotspot", action="store_true", help="Skip HotSpot simulation; write 0.0 placeholder temperatures")
    ap.add_argument("--config_name", type=str, default="reproduce.json")
    ap.add_argument("--config_mode", type=str, default="thermal", choices=["wl", "thermal"])
    ap.add_argument("--unit_scale", type=float, default=0.001)
    ap.add_argument("--hotspot_binary", type=str, default="external/ATPlace_pub/thermal/hotspot")
    ap.add_argument("--grid_size", type=int, nargs=2, default=[64, 64], metavar=("NX", "NY"))
    ap.add_argument("--ambient", type=float, default=25.0)
    ap.add_argument("--scale", type=float, default=0.05)
    ap.add_argument("--sigma_factor", type=float, default=1.0)
    args = ap.parse_args()

    cases = args.cases or discover_cases(args.milp_dir)
    if not cases:
        raise SystemExit(f"No CaseN/summary.json found under {args.milp_dir}")

    thermal_config = {
        "backend": "heuristic" if args.skip_hotspot else "hotspot",
        "hotspot_binary": args.hotspot_binary,
        "hotspot_required": not args.skip_hotspot,
        "grid_size": [int(args.grid_size[0]), int(args.grid_size[1])],
        "ambient": float(args.ambient),
        "scale": float(args.scale),
        "sigma_factor": float(args.sigma_factor),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    thermal_summary: dict[str, dict] = {}
    for case_name in cases:
        summary_path = args.milp_dir / case_name / "summary.json"
        summary = json.loads(summary_path.read_text())

        generator = ThermalDatasetGenerator(
            CASES_DIR / case_name,
            thermal_config,
            config_name=args.config_name,
            config_mode=args.config_mode,
            unit_scale=args.unit_scale,
            work_dir=args.out_dir / "_thermal" / case_name,
        )
        sample = build_sample(generator, summary["chiplets"], args.skip_hotspot)
        if sample is None:
            print(f"[skip] {case_name}: simulation failed")
            continue

        out_file = args.out_dir / case_name / "sample_000000.csv"
        generator.save_sample_format_pointwise_augmented(sample, out_file)
        print(f"[ok] {case_name} -> {out_file}" + ("  (temperature=placeholder)" if args.skip_hotspot else ""))

        if not args.skip_hotspot:
            grid = sample.temperature_map
            grid_file = args.out_dir / case_name / "hotspot_grid.npy"
            np.save(grid_file, grid)
            tmax, t50, tmean, tstd = float(grid.max()), top_k_mean(grid), float(grid.mean()), float(grid.std())
            thermal_summary[case_name] = {
                "grid_file": str(grid_file.relative_to(args.out_dir)),
                "grid_shape": list(grid.shape),
                "tmax_c": tmax,
                "t50_c": t50,
                "tmean_c": tmean,
                "tstd_c": tstd,
            }
            print(f"       Tmax={tmax:.2f}C  T50={t50:.2f}C  Tmean={tmean:.2f}C")

    if thermal_summary:
        summary_file = args.out_dir / "thermal_summary.json"
        summary_file.write_text(json.dumps(thermal_summary, indent=2))
        print(f"\nThermal summary (Tmax/T50/Tmean/Tstd per case) -> {summary_file}")


if __name__ == "__main__":
    main()
