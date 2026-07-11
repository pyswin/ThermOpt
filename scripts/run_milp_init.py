"""
run_milp_init.py — Phase-1 MILP initialisation benchmark (Cases 1-8)

Usage:
    python scripts/run_milp_init.py
    python scripts/run_milp_init.py --time 100
    python scripts/run_milp_init.py --time 150 --spacing 0.05 --epsilon 0.0
    python scripts/run_milp_init.py --cases Case3 Case5 Case6
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
from thermopt.layout.geometry import hpwl, total_overlap_penalty, total_outline_penalty
from thermopt.layout.visualization import draw_layout
from thermopt.optimizer.milp_wl import initialize

CASES_DIR = ROOT / "external/ATPlace_pub/cases"
DEFAULT_CASES = [f"Case{i}" for i in range(1, 9)]


def main(
    time_limit: float = 150.0,
    min_spacing: float = 0.0,
    epsilon: float = 0.0,
    cases: list[str] | None = None,
) -> None:
    if cases is None:
        cases = DEFAULT_CASES

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"t{int(time_limit)}s_sp{min_spacing:.2f}_eps{epsilon:.2f}"
    out_dir = ROOT / "atplace" / "milp_init_runs" / f"{stamp}_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "milp_time_limit": time_limit,
        "mip_rel_gap": 0.001,
        "min_chiplet_spacing": min_spacing,
        "milp_overlap_epsilon": epsilon,
        "verbose": False,
    }

    print(f"\nPhase-1 MILP init  |  time_limit={time_limit}s  spacing={min_spacing}mm  epsilon={epsilon}mm")
    print(f"Output → {out_dir}\n")
    hdr = f"{'Case':8} | {'n':>3} {'nets':>5} | {'WL(m)':>8} {'solver':>6} {'gap':>6} | {'overlap':>9} {'outline':>9} | {'legal':>6} | {'time':>6}"
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
        nets = len(case.nets)

        t0 = time.time()
        result = initialize(case, config)
        elapsed = time.time() - t0

        layout = result.layout
        wl_m = hpwl(case, layout) / 1e3
        overlap = total_overlap_penalty(case, layout)
        outline = total_outline_penalty(case, layout)
        legal = overlap <= 1e-9 and outline <= 1e-9
        obj = result.solver_objective

        status = "OK" if result.solver_success else "FAIL"
        legal_str = "YES" if legal else "NO"

        print(
            f"{case_name:8} | {n:>3} {nets:>5} | {wl_m:>8.3f} {status:>6} {obj:>6.1f} | "
            f"{overlap:>9.4f} {outline:>9.4f} | {legal_str:>6} | {elapsed:>5.0f}s"
        )

        case_out = out_dir / case_name
        case_out.mkdir()

        # Save layout JSON (same format as thermal_runs summaries)
        chiplets_out = [
            {
                "name": p.chiplet_id,
                "cx_mm": round(p.x, 6),
                "cy_mm": round(p.y, 6),
                "x_mm": round(p.x - 0.5 * _rotated_w(case, p), 6),
                "y_mm": round(p.y - 0.5 * _rotated_h(case, p), 6),
                "width_mm": round(_rotated_w(case, p), 6),
                "height_mm": round(_rotated_h(case, p), 6),
                "rotation": p.rotation,
            }
            for p in layout.placements
        ]
        summary = {
            "case": case_name,
            "n_chiplets": n,
            "n_nets": nets,
            "solver_success": result.solver_success,
            "solver_message": result.solver_message,
            "solver_objective": float(obj),
            "wl_m": wl_m,
            "overlap_penalty": overlap,
            "outline_penalty": outline,
            "legal": legal,
            "runtime_s": elapsed,
            "config": config,
            "chiplets": chiplets_out,
        }
        (case_out / "summary.json").write_text(json.dumps(summary, indent=2))

        # Visualise layout
        fig, ax = plt.subplots(figsize=(7, 7))
        draw_layout(ax, case, layout, f"{case_name}  WL={wl_m:.3f}m  {'LEGAL' if legal else 'ILLEGAL'}")
        _annotate_legality(ax, overlap, outline)
        fig.tight_layout()
        fig.savefig(case_out / "layout.png", dpi=160)
        plt.close(fig)

        all_results[case_name] = summary

    # Overview grid
    n_cases = len(all_results)
    if n_cases > 0:
        cols = min(4, n_cases)
        rows = (n_cases + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
        axes_flat = [axes] if n_cases == 1 else (axes.flat if rows > 1 else axes)
        for ax, (case_name, summary) in zip(axes_flat, all_results.items()):
            ci = load_atplace_case(CASES_DIR / case_name, {}, seed=42)
            result_layout = _layout_from_summary(ci.case, summary)
            wl = summary["wl_m"]
            legal_str = "LEGAL" if summary["legal"] else "ILLEGAL"
            draw_layout(ax, ci.case, result_layout, f"{case_name}  {wl:.3f}m  {legal_str}")
        for ax in list(axes_flat)[n_cases:]:
            ax.set_visible(False)
        fig.suptitle(f"Phase-1 MILP Init  |  {tag}", fontsize=13)
        fig.tight_layout()
        fig.savefig(out_dir / "overview.png", dpi=140)
        plt.close(fig)

    total_t = time.time() - t_total
    json.dump({"config": config, "cases": all_results}, (out_dir / "all_results.json").open("w"), indent=2)

    legal_count = sum(1 for r in all_results.values() if r["legal"])
    print(f"\nLegal: {legal_count}/{len(all_results)}  |  total {total_t/60:.1f} min  |  {out_dir}")


# ── helpers ──────────────────────────────────────────────────────────────────

def _rotated_w(case, placement):
    w, h = placement.rotated_size(case.chiplet_by_id[placement.chiplet_id])
    return w


def _rotated_h(case, placement):
    w, h = placement.rotated_size(case.chiplet_by_id[placement.chiplet_id])
    return h


def _annotate_legality(ax, overlap, outline):
    if overlap > 1e-9 or outline > 1e-9:
        ax.text(
            0.02, 0.98, f"overlap={overlap:.4f}  outline={outline:.4f}",
            transform=ax.transAxes, va="top", fontsize=7, color="red",
        )


def _layout_from_summary(case, summary):
    from thermopt.layout.objects import Layout, Placement
    placements = [
        Placement(c["name"], c["cx_mm"], c["cy_mm"], c["rotation"])
        for c in summary["chiplets"]
    ]
    return Layout(tuple(placements))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", type=float, default=150.0, help="MILP time limit per case (s)")
    parser.add_argument("--spacing", type=float, default=0.0, help="min chiplet spacing (mm)")
    parser.add_argument("--epsilon", type=float, default=0.0, help="overlap relaxation (mm)")
    parser.add_argument("--cases", nargs="+", default=None)
    args = parser.parse_args()
    main(time_limit=args.time, min_spacing=args.spacing, epsilon=args.epsilon, cases=args.cases)
