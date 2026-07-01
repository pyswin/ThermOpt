"""
UFNO (ScOT) vs HotSpot accuracy check on initial MILP layouts.
Uses cached HotSpot 2D grids (hs key from grid_cache_init.npz),
re-computes UFNO predictions with current code.
Layout: 5 rows (cases) x 3 cols [HotSpot | UFNO | Error (HS-UFNO)]
"""
import sys, json, math
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import zoom

UFNO_DEMO = Path(__file__).parent.parent / "src/thermopt/thermal/ufno_demo"
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(UFNO_DEMO))

from thermopt.data.atplace import load_atplace_case
from thermopt.layout.objects import Layout, Placement
from thermopt.thermal.surrogate_input import coordinate_grid, rasterize_power_channel, SURROGATE_NATIVE_GRID_SIZE

SCALE     = 0.001
CASES_DIR = Path(__file__).parent.parent / "external/ATPlace_pub/cases"
MILP_DIR  = Path(__file__).parent.parent / "atplace/20260627_163106_milp150s"
MODEL_PT  = str(UFNO_DEMO / "model.pt")
CACHE_DIR = Path(__file__).parent.parent / "atplace/thermal_runs/20260628_004144_maxT"
OUT_FILE  = CACHE_DIR / "ufno_accuracy.png"

CASES = ["Case3", "Case5", "Case6", "Case7", "Case8"]


def load_milp_layout(case_name):
    d = json.load(open(MILP_DIR / case_name / "layout.json"))
    return Layout(placements=[
        Placement(chiplet_id=c["name"],
                  x=c["x"] * SCALE, y=c["y"] * SCALE,
                  rotation=int(round(math.degrees(c["angle_rad"]))) % 360)
        for c in d["chiplets"]
    ])


def ufno_predict(case, layout, x_norm, model, y_norm):
    """Run UFNO inference: returns (64,64) temperature in Celsius."""
    gx, gy = coordinate_grid(case, SURROGATE_NATIVE_GRID_SIZE)
    power = np.zeros((64, 64), dtype=np.float32)
    rasterize_power_channel(case, layout, gx, gy, out=power)
    # UFNO input shape: (64, 64, 1, 3) = [rows=Y, cols=X, Z=1, (power, gx, gy)]
    x_phys = np.stack([power, gx, gy], axis=-1)[:, :, np.newaxis, :]
    xt = torch.from_numpy(x_phys).unsqueeze(0)   # (1, 64, 64, 1, 3)
    xn = x_norm.forward(xt)
    with torch.no_grad():
        out = model(xn)
    pred_k = y_norm.inverse(out)[0, :, :, 0].numpy()  # (64, 64) Kelvin
    return (pred_k - 273.15).astype(np.float32)


def main():
    # Load UFNO model once
    x_norm, model, y_norm = torch.load(MODEL_PT, map_location="cpu", weights_only=False)
    model.eval()
    print(f"Loaded UFNO ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")

    fig, axes = plt.subplots(len(CASES), 3,
                             figsize=(13, 4.0 * len(CASES)),
                             constrained_layout=False)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.93, bottom=0.04,
                        hspace=0.5, wspace=0.4)

    col_titles = ["HotSpot (ground truth)", "UFNO prediction", "Error (HotSpot − UFNO)"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=11, pad=6)

    summary = []

    for i, case_name in enumerate(CASES):
        print(f"[{i+1}/{len(CASES)}] {case_name} ...", flush=True)
        ci = load_atplace_case(CASES_DIR / case_name, {}, 42)
        case = ci.case
        W, H = case.outline_width, case.outline_height
        extent = [0, W, 0, H]

        # Load HotSpot 2D grid from cache (hs key is always valid)
        cache_path = CACHE_DIR / case_name / "grid_cache_init.npz"
        if not cache_path.exists():
            print(f"  [SKIP] no HotSpot cache for {case_name}", flush=True)
            continue
        hs = np.load(cache_path)["hs"]

        # UFNO prediction
        layout = load_milp_layout(case_name)
        ufno = ufno_predict(case, layout, x_norm, model, y_norm)

        # Resample UFNO to match HotSpot grid size for error map
        zy = hs.shape[0] / ufno.shape[0]
        zx = hs.shape[1] / ufno.shape[1]
        ufno_r = zoom(ufno, (zy, zx), order=1)

        err = hs - ufno_r

        hs_tmax   = float(hs.max())
        ufno_tmax = float(ufno.max())
        flat_hs   = sorted(hs.flatten(), reverse=True)
        flat_u    = sorted(ufno.flatten(), reverse=True)
        hs_t50    = float(sum(flat_hs[:50]) / 50)
        ufno_t50  = float(sum(flat_u[:50]) / 50)
        summary.append((case_name, hs_tmax, ufno_tmax, hs_t50, ufno_t50))

        print(f"  HotSpot Tmax={hs_tmax:.1f}  UFNO Tmax={ufno_tmax:.1f}  err={ufno_tmax-hs_tmax:+.1f}°C", flush=True)

        vmin = min(hs.min(), ufno_r.min())
        vmax = max(hs.max(), ufno_r.max())
        eabs = max(abs(err).max(), 1.0)

        maps  = [hs, ufno_r, err]
        cmaps = ["hot", "hot", "RdBu_r"]
        vmins = [vmin, vmin, -eabs]
        vmaxs = [vmax, vmax,  eabs]

        for j, (arr, cmap, mn, mx) in enumerate(zip(maps, cmaps, vmins, vmaxs)):
            ax = axes[i, j]
            im = ax.imshow(arr, origin="lower", extent=extent,
                           cmap=cmap, vmin=mn, vmax=mx, aspect="auto",
                           interpolation="bicubic")
            ax.set_xlabel("x (mm)", fontsize=7)
            ax.set_ylabel("y (mm)", fontsize=7)
            ax.tick_params(labelsize=6)
            ax.xaxis.set_major_locator(mticker.MaxNLocator(4, integer=True))
            ax.yaxis.set_major_locator(mticker.MaxNLocator(4, integer=True))

            if j == 0:
                ax.text(-0.38, 0.5,
                        f"{case_name}\n({W:.0f}x{H:.0f} mm)",
                        transform=ax.transAxes, fontsize=9, fontweight="bold",
                        va="center", ha="center", rotation=90)
                ax.set_title(f"Tmax={hs_tmax:.1f}°C", fontsize=7, pad=2, color="#333")
            if j == 1:
                sign = "+" if ufno_tmax >= hs_tmax else ""
                ax.set_title(f"Tmax={ufno_tmax:.1f}°C  (err={sign}{ufno_tmax-hs_tmax:.1f}°C)",
                             fontsize=7, pad=2, color="#333")

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.04)
            cb = fig.colorbar(im, cax=cax)
            cb.ax.tick_params(labelsize=6)
            cb.set_label("°C (diff)" if j == 2 else "°C", fontsize=6)

    fig.suptitle("UFNO (ScOT) vs HotSpot — Initial MILP Layouts", fontsize=13, y=0.97)
    fig.savefig(OUT_FILE, dpi=150, bbox_inches="tight")
    print(f"\nSaved -> {OUT_FILE}")

    print(f"\n{'Case':7} | {'HS_Tmax':>9} {'UFNO_Tmax':>10} {'err_Tmax':>9} | {'HS_T50':>8} {'UFNO_T50':>9} {'err_T50':>8}")
    print("-" * 68)
    for case_name, hs_tmax, u_tmax, hs_t50, u_t50 in summary:
        print(f"{case_name:7} | {hs_tmax:>9.2f} {u_tmax:>10.2f} {u_tmax-hs_tmax:>+9.2f} | "
              f"{hs_t50:>8.2f} {u_t50:>9.2f} {u_t50-hs_t50:>+8.2f}")


if __name__ == "__main__":
    main()
