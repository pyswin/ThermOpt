"""
HotSpot vs ScOT temperature map comparison (tmax objective).
Layout: 5 rows (Case3/5/6/7/8) x 6 columns
  Left 3:  Initial layout  [HotSpot | ScOT | Error (HS-ScOT)]
  Right 3: Optimized layout [HotSpot | ScOT | Error (HS-ScOT)]
"""
import sys, json, math, time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from thermopt.data.atplace import load_atplace_case
from thermopt.layout.objects import Layout, Placement
from thermopt.thermal.hotspot import HotSpotBackend
from thermopt.thermal.thermfm import ThermFMThermalBackend

SCALE     = 0.001
CASES_DIR = Path(__file__).parent.parent / "external/ATPlace_pub/cases"
MILP_DIR  = Path(__file__).parent.parent / "atplace/20260627_163106_milp150s"
MODEL_DIR = str(Path(__file__).parent.parent / "src/thermopt/thermal/thermfm_t_case_all_demo/model")
HOTSPOT_B = str(Path(__file__).parent.parent / "external/ATPlace_pub/thermal/hotspot")
RUN_DIR   = Path(__file__).parent.parent / "atplace/thermal_runs/20260628_004144_maxT"
OUT_FILE  = RUN_DIR / "thermal_comparison.png"

CASES     = ["Case3", "Case5", "Case6", "Case7", "Case8"]
GRID_SIZE = (80, 100)  # rows, cols for HotSpot (will resample ScOT to match)
MODE      = "tmax"


def load_layout_milp(case_name):
    d = json.load(open(MILP_DIR / case_name / "layout.json"))
    return Layout(placements=[
        Placement(chiplet_id=c["name"],
                  x=c["x"] * SCALE, y=c["y"] * SCALE,
                  rotation=int(round(math.degrees(c["angle_rad"]))) % 360)
        for c in d["chiplets"]
    ])


def load_layout_opt(case_name):
    d = json.load(open(RUN_DIR / case_name / MODE / "summary.json"))
    return Layout(placements=[
        Placement(chiplet_id=c["name"],
                  x=c["x_mm"], y=c["y_mm"],
                  rotation=c["rotation"])
        for c in d["chiplets"]
    ])


def get_grids(case, layout, hs_backend, scot_backend, cache_path=None):
    if cache_path and cache_path.exists():
        d = np.load(cache_path)
        return d["hs"], d["scot_r"]
    hs   = hs_backend.simulate(case, layout)
    scot = scot_backend.simulate(case, layout)
    from scipy.ndimage import zoom
    zy = hs.shape[0] / scot.shape[0]
    zx = hs.shape[1] / scot.shape[1]
    scot_r = zoom(scot, (zy, zx), order=1)
    if cache_path:
        np.savez(cache_path, hs=hs, scot_r=scot_r)
    return hs, scot_r


def main():
    fig, axes = plt.subplots(
        len(CASES), 6,
        figsize=(22, 4.2 * len(CASES)),
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.06, right=0.97, top=0.94, bottom=0.04,
                        hspace=0.45, wspace=0.35)

    col_titles = [
        "Initial — HotSpot", "Initial — ScOT", "Initial — Error (HS−ScOT)",
        "Optimized — HotSpot", "Optimized — ScOT", "Optimized — Error (HS−ScOT)",
    ]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=10, pad=6)

    for i, case_name in enumerate(CASES):
        print(f"[{i+1}/{len(CASES)}] {case_name} ...", flush=True)
        ci = load_atplace_case(CASES_DIR / case_name, {}, 42)
        case = ci.case
        W = case.outline_width   # mm
        H = case.outline_height  # mm
        extent = [0, W, 0, H]

        hs_backend   = HotSpotBackend(case=case,
                                      config={"hotspot_binary": HOTSPOT_B,
                                              "grid_size": list(GRID_SIZE)})
        scot_backend = ThermFMThermalBackend(
            case=case,
            config={"backend": "thermfm", "thermfm_model_dir": MODEL_DIR})

        layout_init = load_layout_milp(case_name)
        layout_opt  = load_layout_opt(case_name)

        cache_dir = RUN_DIR / case_name
        t0 = time.time()
        hs_i,  scot_i  = get_grids(case, layout_init, hs_backend, scot_backend,
                                    cache_dir / "grid_cache_init.npz")
        hs_o,  scot_o  = get_grids(case, layout_opt,  hs_backend, scot_backend,
                                    cache_dir / f"grid_cache_{MODE}.npz")
        print(f"  grids done ({time.time()-t0:.0f}s)", flush=True)

        err_i = hs_i - scot_i
        err_o = hs_o - scot_o

        # Color range: share vmin/vmax across all four temp maps per case
        vmin = min(hs_i.min(), scot_i.min(), hs_o.min(), scot_o.min())
        vmax = max(hs_i.max(), scot_i.max(), hs_o.max(), scot_o.max())
        # Error colormap: symmetric around 0
        eabs = max(abs(err_i).max(), abs(err_o).max())

        maps = [hs_i, scot_i, err_i, hs_o, scot_o, err_o]
        cmaps = ["hot", "hot", "RdBu_r", "hot", "hot", "RdBu_r"]
        vmins = [vmin, vmin, -eabs, vmin, vmin, -eabs]
        vmaxs = [vmax, vmax,  eabs, vmax, vmax,  eabs]

        for j, (arr, cmap, mn, mx) in enumerate(zip(maps, cmaps, vmins, vmaxs)):
            ax = axes[i, j]
            im = ax.imshow(arr, origin="lower", extent=extent,
                           cmap=cmap, vmin=mn, vmax=mx, aspect="auto",
                           interpolation="bicubic")
            ax.set_xlabel("x (mm)", fontsize=7)
            ax.set_ylabel("y (mm)", fontsize=7)
            if j == 0:
                ax.text(-0.32, 0.5, f"{case_name}\n({H:.0f}x{W:.0f} mm)",
                        transform=ax.transAxes, fontsize=9, fontweight="bold",
                        va="center", ha="center", rotation=90)
            ax.tick_params(labelsize=6)
            ax.xaxis.set_major_locator(mticker.MaxNLocator(4, integer=True))
            ax.yaxis.set_major_locator(mticker.MaxNLocator(4, integer=True))

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.04)
            cb = fig.colorbar(im, cax=cax)
            cb.ax.tick_params(labelsize=6)
            unit = "°C (diff)" if j in (2, 5) else "°C"
            cb.set_label(unit, fontsize=6)

            # Annotate peak temps
            if j in (0, 3):
                ax.set_title(
                    f"Tmax={arr.max():.1f}°C  T50={sorted(arr.flatten())[-50:][0]:.1f}°C",
                    fontsize=6.5, pad=2, color="#333")

        print(f"  {case_name} plotted", flush=True)

    fig.suptitle(
        "HotSpot vs ScOT Temperature Map Comparison  [tw=10000, tmax objective]",
        fontsize=13, y=0.97,
    )
    fig.savefig(OUT_FILE, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {OUT_FILE}")


if __name__ == "__main__":
    main()
