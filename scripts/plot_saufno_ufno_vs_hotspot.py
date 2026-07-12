"""
HotSpot (ground truth) vs SAU-FNO vs U-FNO temperature map comparison, on the
MILP layouts found under a given --milp_dir.

Adapted from plot_thermal_comparison.py / plot_ufno_accuracy.py: reuses their
column-grid plotting layout, but drives the two surrogate models through the
real ThermalBackend classes (UFNOThermalBackend / SAUFNOThermalBackend)
instead of ad-hoc torch loading.

Two known --milp_dir layouts are auto-detected per case by which file exists,
since different historical runs used different coordinate conventions --
verified empirically each time (interpreting as center vs lower-left corner,
whichever gives zero overlap and no outline violation), not assumed:
  - CaseN/layout.json  (e.g. atplace/20260627_163106_milp150s): external
    ATPlace_pub/reproduce.py output. Units um, x/y is CENTER, rotation as
    angle_rad.
  - CaseN/summary.json (e.g. atplace/milp_runs/20260712_case1_10_gurobi_hotspot): this repo's
    own MILP+HotSpot scripts. Units mm, x_mm/y_mm is the lower-left CORNER
    (same historical convention scripts/hotspot_validate.py's
    load_layout_from_summary already converts), rotation in degrees.

Layout: N rows (one per case found) x 5 cols:
  HotSpot | SAU-FNO | Error (HS-SAUFNO) | U-FNO | Error (HS-UFNO)
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from thermopt.data.atplace import load_atplace_case
from thermopt.layout.objects import FloorplanCase, Layout, Placement
from thermopt.thermal.hotspot import HotSpotBackend
from thermopt.thermal.saufno import SAUFNOThermalBackend
from thermopt.thermal.ufno import UFNOThermalBackend

SCALE = 0.001  # layout.json (reproduce.py) is in um; thermopt works in mm
ROOT = Path(__file__).parent.parent
CASES_DIR = ROOT / "external/ATPlace_pub/cases"
HOTSPOT_BIN = str(ROOT / "external/ATPlace_pub/thermal/hotspot")
GRID_SIZE = [64, 64]  # shared by all three backends -> no resampling needed
TOP_K = 50


def top_k_mean(grid: np.ndarray, k: int = TOP_K) -> float:
    """Mean of the k hottest grid cells -- a hotspot-region metric that's more
    robust than a single-pixel Tmax (matches plot_ufno_accuracy.py's T50)."""
    flat = grid.flatten()
    k = min(k, flat.size)
    return float(np.mean(np.sort(flat)[-k:]))


def _discover_cases(milp_dir: Path) -> list[str]:
    return sorted(
        (
            p.name for p in milp_dir.iterdir()
            if p.is_dir() and ((p / "layout.json").exists() or (p / "summary.json").exists())
        ),
        key=lambda name: int("".join(ch for ch in name if ch.isdigit()) or 0),
    )


def load_layout_reproduce_json(case_dir: Path) -> Layout:
    """CaseN/layout.json (external/ATPlace_pub/reproduce.py): um, x/y is center."""
    d = json.load(open(case_dir / "layout.json"))
    return Layout(placements=tuple(
        Placement(
            chiplet_id=c["name"],
            x=c["x"] * SCALE,
            y=c["y"] * SCALE,
            rotation=int(round(math.degrees(c["angle_rad"]))) % 360,
        )
        for c in d["chiplets"]
    ))


def load_layout_from_summary(case: FloorplanCase, case_dir: Path) -> Layout:
    """CaseN/summary.json: mm, x_mm/y_mm is the lower-left corner (matches
    scripts/hotspot_validate.py's load_layout_from_summary)."""
    d = json.load(open(case_dir / "summary.json"))
    placements = []
    for c in d["chiplets"]:
        chiplet = case.chiplet_by_id[c["name"]]
        width, height = (chiplet.height, chiplet.width) if c["rotation"] % 180 == 90 else (chiplet.width, chiplet.height)
        placements.append(Placement(
            chiplet_id=c["name"],
            x=c["x_mm"] + width * 0.5,
            y=c["y_mm"] + height * 0.5,
            rotation=c["rotation"],
        ))
    return Layout(placements=placements)


def load_layout(case: FloorplanCase, case_dir: Path) -> Layout:
    if (case_dir / "layout.json").exists():
        return load_layout_reproduce_json(case_dir)
    if (case_dir / "summary.json").exists():
        return load_layout_from_summary(case, case_dir)
    raise FileNotFoundError(f"no layout.json or summary.json under {case_dir}")


def get_grid(case, layout, backend, cache_path: Path) -> np.ndarray:
    if cache_path.exists():
        return np.load(cache_path)["grid"]
    grid = backend.simulate(case, layout)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, grid=grid)
    return grid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--milp_dir", type=Path, required=True, help="Directory containing CaseN/{layout,summary}.json")
    parser.add_argument("--cases", nargs="+", default=None, help="Override auto-discovered case list")
    parser.add_argument("--out_dir", type=Path, default=None, help="Default: atplace/thermal_runs/<milp_dir name>_vs_hotspot")
    args = parser.parse_args()

    milp_dir = args.milp_dir.resolve()
    cases = args.cases or _discover_cases(milp_dir)
    if not cases:
        raise SystemExit(f"no CaseN/{{layout,summary}}.json found under {milp_dir}")
    out_dir = args.out_dir or (ROOT / "atplace/thermal_runs" / f"{milp_dir.name}_vs_hotspot")
    out_file = out_dir / "saufno_ufno_vs_hotspot.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"milp_dir={milp_dir}  cases={cases}  out_dir={out_dir}", flush=True)

    fig, axes = plt.subplots(len(cases), 5, figsize=(18, 3.6 * len(cases)), constrained_layout=False)
    if len(cases) == 1:
        axes = axes.reshape(1, -1)
    fig.subplots_adjust(left=0.06, right=0.97, top=0.96, bottom=0.03, hspace=0.5, wspace=0.4)

    col_titles = ["HotSpot (ground truth)", "SAU-FNO", "Error (HS-SAUFNO)", "U-FNO", "Error (HS-UFNO)"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=11, pad=6)

    summary = []
    t_total = time.time()
    for i, case_name in enumerate(cases):
        print(f"[{i + 1}/{len(cases)}] {case_name} ...", flush=True)
        ci = load_atplace_case(CASES_DIR / case_name, {}, seed=42)
        case = ci.case
        W, H = case.outline_width, case.outline_height
        extent = [0, W, 0, H]
        layout = load_layout(case, milp_dir / case_name)

        hotspot = HotSpotBackend(case=case, config={"hotspot_binary": HOTSPOT_BIN, "grid_size": GRID_SIZE})
        saufno = SAUFNOThermalBackend(case=case, config={"backend": "saufno", "grid_size": GRID_SIZE})
        ufno = UFNOThermalBackend(case=case, config={"backend": "ufno", "grid_size": GRID_SIZE})

        cache_dir = out_dir / case_name
        t0 = time.time()
        hs = get_grid(case, layout, hotspot, cache_dir / "hotspot.npz")
        sau = get_grid(case, layout, saufno, cache_dir / "saufno.npz")
        uf = get_grid(case, layout, ufno, cache_dir / "ufno.npz")
        print(f"  grids done ({time.time() - t0:.0f}s)", flush=True)

        err_sau = hs - sau
        err_uf = hs - uf

        hs_tmax, sau_tmax, uf_tmax = float(hs.max()), float(sau.max()), float(uf.max())
        hs_t50, sau_t50, uf_t50 = top_k_mean(hs), top_k_mean(sau), top_k_mean(uf)
        summary.append((
            case_name, hs_tmax, sau_tmax, sau_tmax - hs_tmax, uf_tmax, uf_tmax - hs_tmax,
            hs_t50, sau_t50, sau_t50 - hs_t50, uf_t50, uf_t50 - hs_t50,
        ))
        print(
            f"  HotSpot Tmax={hs_tmax:.1f}  SAU-FNO Tmax={sau_tmax:.1f} (err={sau_tmax - hs_tmax:+.1f}°C)  "
            f"U-FNO Tmax={uf_tmax:.1f} (err={uf_tmax - hs_tmax:+.1f}°C)",
            flush=True,
        )
        print(
            f"  HotSpot T{TOP_K}={hs_t50:.1f}  SAU-FNO T{TOP_K}={sau_t50:.1f} (err={sau_t50 - hs_t50:+.1f}°C)  "
            f"U-FNO T{TOP_K}={uf_t50:.1f} (err={uf_t50 - hs_t50:+.1f}°C)",
            flush=True,
        )

        vmin = min(hs.min(), sau.min(), uf.min())
        vmax = max(hs.max(), sau.max(), uf.max())
        eabs = max(abs(err_sau).max(), abs(err_uf).max(), 1.0)

        maps = [hs, sau, err_sau, uf, err_uf]
        cmaps = ["hot", "hot", "RdBu_r", "hot", "RdBu_r"]
        vmins = [vmin, vmin, -eabs, vmin, -eabs]
        vmaxs = [vmax, vmax, eabs, vmax, eabs]
        titles = [
            f"Tmax={hs_tmax:.1f}°C  T{TOP_K}={hs_t50:.1f}°C",
            f"Tmax={sau_tmax:.1f}°C (err={sau_tmax - hs_tmax:+.1f})  T{TOP_K} err={sau_t50 - hs_t50:+.1f}°C",
            None,
            f"Tmax={uf_tmax:.1f}°C (err={uf_tmax - hs_tmax:+.1f})  T{TOP_K} err={uf_t50 - hs_t50:+.1f}°C",
            None,
        ]

        for j, (arr, cmap, mn, mx, title) in enumerate(zip(maps, cmaps, vmins, vmaxs, titles)):
            ax = axes[i, j]
            im = ax.imshow(arr, origin="lower", extent=extent, cmap=cmap, vmin=mn, vmax=mx, aspect="auto", interpolation="bicubic")
            ax.set_xlabel("x (mm)", fontsize=7)
            ax.set_ylabel("y (mm)", fontsize=7)
            ax.tick_params(labelsize=6)
            ax.xaxis.set_major_locator(mticker.MaxNLocator(4, integer=True))
            ax.yaxis.set_major_locator(mticker.MaxNLocator(4, integer=True))
            if title:
                ax.set_title(title, fontsize=7, pad=2, color="#333")
            if j == 0:
                ax.text(
                    -0.38, 0.5, f"{case_name}\n({W:.0f}x{H:.0f} mm)",
                    transform=ax.transAxes, fontsize=9, fontweight="bold",
                    va="center", ha="center", rotation=90,
                )

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.04)
            cb = fig.colorbar(im, cax=cax)
            cb.ax.tick_params(labelsize=6)
            cb.set_label("°C (diff)" if j in (2, 4) else "°C", fontsize=6)

    fig.suptitle(f"SAU-FNO / U-FNO vs HotSpot — MILP Layouts ({milp_dir.name})", fontsize=13, y=0.995)
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    print(f"\nSaved -> {out_file}")

    hdr = (
        f"{'Case':7} | {'HS_Tmax':>8} {'SAU_Tmax':>8} {'SAU_err':>7} | {'UF_Tmax':>8} {'UF_err':>7} || "
        f"{'HS_T'+str(TOP_K):>8} {'SAU_T'+str(TOP_K):>8} {'SAU_err':>7} | {'UF_T'+str(TOP_K):>8} {'UF_err':>7}"
    )
    print(f"\n{hdr}")
    print("-" * len(hdr))
    for case_name, hs_tmax, sau_tmax, sau_err, uf_tmax, uf_err, hs_t50, sau_t50, sau_t50_err, uf_t50, uf_t50_err in summary:
        print(
            f"{case_name:7} | {hs_tmax:>8.2f} {sau_tmax:>8.2f} {sau_err:>+7.2f} | {uf_tmax:>8.2f} {uf_err:>+7.2f} || "
            f"{hs_t50:>8.2f} {sau_t50:>8.2f} {sau_t50_err:>+7.2f} | {uf_t50:>8.2f} {uf_t50_err:>+7.2f}"
        )

    (out_dir / "summary.json").write_text(json.dumps(
        [
            {
                "case": c, "hotspot_tmax": ht, "saufno_tmax": st, "saufno_err": se, "ufno_tmax": ut, "ufno_err": ue,
                f"hotspot_t{TOP_K}": ht50, f"saufno_t{TOP_K}": st50, f"saufno_t{TOP_K}_err": st50e,
                f"ufno_t{TOP_K}": ut50, f"ufno_t{TOP_K}_err": ut50e,
            }
            for c, ht, st, se, ut, ue, ht50, st50, st50e, ut50, ut50e in summary
        ],
        indent=2,
    ))
    print(f"\nAll done in {(time.time() - t_total) / 60:.1f} min.")


if __name__ == "__main__":
    main()
