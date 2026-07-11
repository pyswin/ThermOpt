"""
run_milp150s_hotspot.py — Gurobi Phase-1 MILP (150s) + WL + HotSpot initial Tmax
for Cases 1-8, for comparison against the original ATPlace paper's numbers.

Only the plain MILP result is used (no snapshots, no perturbation). Per case:
  - solve Phase-1 MILP init with Gurobi (milp_time_limit, default 150s)
  - HPWL of the resulting layout
  - HotSpot ground-truth simulation of that layout (Tmax, top5% mean)

Usage (needs gurobipy + a valid license, e.g. conda env atplace_py39):
    /path/to/envs/atplace_py39/bin/python scripts/run_milp150s_hotspot.py
    /path/to/envs/atplace_py39/bin/python scripts/run_milp150s_hotspot.py --time 100 --cases Case3 Case5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from thermopt.data.atplace import load_atplace_case
from thermopt.layout.geometry import hpwl, total_outline_penalty, total_overlap_penalty
from thermopt.layout.visualization import draw_layout, save_temperature_figure
from thermopt.optimizer.milp_wl_gurobi import initialize_with_snapshots
from thermopt.thermal.hotspot import HotSpotBackend

CASES_DIR = ROOT / "external/ATPlace_pub/cases"
HS_BIN = ROOT / "external/ATPlace_pub/thermal/hotspot"
DEFAULT_CASES = [f"Case{i}" for i in range(1, 9)]


def main(time_limit: float = 150.0, cases: list[str] | None = None) -> None:
    cases = cases or DEFAULT_CASES

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "atplace" / "milp_runs" / f"{stamp}_gurobi_t{int(time_limit)}s_hotspot"
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {"milp_time_limit": time_limit, "mip_rel_gap": 0.001, "snapshot_interval": 1e9, "verbose": False}

    print(f"\nGurobi MILP init (t={time_limit}s) + HotSpot initial Tmax  |  cases={cases}")
    print(f"Output -> {out_dir}\n")
    hdr = f"{'Case':8} | {'n':>3} | {'WL(m)':>8} {'legal':>6} {'MILPt':>6} | {'HS_Tmax':>8} {'HS_top5':>8} {'HSt':>6}"
    print(hdr)
    print("-" * len(hdr))

    all_results: dict[str, dict] = {}
    t_total = time.time()

    for case_name in cases:
        case_dir = CASES_DIR / case_name
        if not case_dir.exists():
            print(f"{case_name:8} | SKIP (directory not found)")
            continue

        ci = load_atplace_case(case_dir, {}, seed=42)
        case = ci.case
        n = len(case.chiplets)

        t0 = time.time()
        result, _snapshots = initialize_with_snapshots(case, config)
        milp_t = time.time() - t0

        layout = result.layout
        wl_m = hpwl(case, layout) / 1e3
        overlap = total_overlap_penalty(case, layout)
        outline = total_outline_penalty(case, layout)
        legal = overlap <= 1e-9 and outline <= 1e-9

        t0 = time.time()
        backend = HotSpotBackend(case=case, config={"hotspot_binary": str(HS_BIN)})
        grid = backend.simulate(case, layout)
        hs_t = time.time() - t0
        flat_sorted = sorted(grid.flatten(), reverse=True)
        hs_tmax = flat_sorted[0]
        hs_top5 = sum(flat_sorted[: max(1, len(flat_sorted) // 20)]) / max(1, len(flat_sorted) // 20)

        print(
            f"{case_name:8} | {n:>3} | {wl_m:>8.3f} {'YES' if legal else 'NO':>6} {milp_t:>5.0f}s | "
            f"{hs_tmax:>8.2f} {hs_top5:>8.2f} {hs_t:>5.0f}s"
        )

        case_out = out_dir / case_name
        case_out.mkdir()
        chiplets_out = []
        for p in layout.placements:
            w, h = p.rotated_size(case.chiplet_by_id[p.chiplet_id])
            chiplets_out.append(
                {
                    "name": p.chiplet_id,
                    "x_mm": round(p.x - 0.5 * w, 6),
                    "y_mm": round(p.y - 0.5 * h, 6),
                    "cx_mm": round(p.x, 6),
                    "cy_mm": round(p.y, 6),
                    "rotation": p.rotation,
                }
            )
        summary = {
            "case": case_name,
            "n_chiplets": n,
            "solver_success": result.solver_success,
            "solver_message": result.solver_message,
            "solver_objective": float(result.solver_objective),
            "milp_time_limit_s": time_limit,
            "milp_runtime_s": milp_t,
            "wl_m": wl_m,
            "overlap_penalty": overlap,
            "outline_penalty": outline,
            "legal": legal,
            "hotspot_tmax_c": hs_tmax,
            "hotspot_top5pct_c": hs_top5,
            "hotspot_runtime_s": hs_t,
            "chiplets": chiplets_out,
        }
        (case_out / "summary.json").write_text(json.dumps(summary, indent=2))

        fig, ax = plt.subplots(figsize=(7, 7))
        draw_layout(ax, case, layout, f"{case_name}  WL={wl_m:.3f}m  Tmax={hs_tmax:.1f}C  {'LEGAL' if legal else 'ILLEGAL'}")
        fig.tight_layout()
        fig.savefig(case_out / "layout.png", dpi=160)
        plt.close(fig)
        save_temperature_figure(grid, case_out / "hotspot_temp.png", f"{case_name}  HotSpot Tmax={hs_tmax:.1f}C")

        all_results[case_name] = summary

    total_t = time.time() - t_total
    (out_dir / "all_results.json").write_text(json.dumps({"config": config, "cases": all_results}, indent=2))
    legal_count = sum(1 for r in all_results.values() if r["legal"])
    print(f"\nLegal: {legal_count}/{len(all_results)}  |  total {total_t/60:.1f} min  |  {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", type=float, default=150.0, help="MILP time limit per case (s)")
    parser.add_argument("--cases", nargs="+", default=None)
    args = parser.parse_args()
    main(time_limit=args.time, cases=args.cases)
