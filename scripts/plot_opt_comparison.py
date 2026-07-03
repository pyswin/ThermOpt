"""
优化前后对比图：HotSpot + UFNO预测 + Error
格式：5 case × 6列 [初始:HS|UFNO|Err]  [优化后:HS|UFNO|Err]
用法:
    python3 scripts/plot_opt_comparison.py --run_dir atplace/thermal_runs/20260628_xxx_ufno_tw50000
"""
import argparse, json, math, sys, time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import zoom

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/thermopt/thermal/ufno_demo"))

from thermopt.data.atplace import load_atplace_case
from thermopt.layout.objects import Layout, Placement
from thermopt.thermal.surrogate_input import (
    coordinate_grid, rasterize_power_channel, SURROGATE_NATIVE_GRID_SIZE,
)
from thermopt.thermal.hotspot import HotSpotBackend

SCALE     = 0.001
CASES_DIR = ROOT / "external/ATPlace_pub/cases"
MILP_DIR  = ROOT / "atplace/20260627_163106_milp150s"
UFNO_PT   = ROOT / "src/thermopt/thermal/ufno_demo/model.pt"
HS_BIN    = ROOT / "external/ATPlace_pub/thermal/hotspot"
CASES     = ["Case3", "Case5", "Case6", "Case7", "Case8"]


_UFNO = {}
def get_ufno():
    if not _UFNO:
        xn, m, yn = torch.load(str(UFNO_PT), map_location="cpu", weights_only=False)
        m.eval()
        _UFNO["state"] = (xn, m, yn)
        print(f"[UFNO] loaded ({sum(p.numel() for p in m.parameters())/1e6:.1f}M params)")
    return _UFNO["state"]


def ufno_predict(case, layout) -> np.ndarray:
    x_norm, model, y_norm = get_ufno()
    gx, gy = coordinate_grid(case, SURROGATE_NATIVE_GRID_SIZE)
    power  = np.zeros((64, 64), dtype=np.float32)
    rasterize_power_channel(case, layout, gx, gy, out=power)
    x_phys = np.stack([power, gx, gy], axis=-1)[:, :, np.newaxis, :]
    xt = torch.from_numpy(x_phys).unsqueeze(0)
    with torch.no_grad():
        pred_k = y_norm.inverse(model(x_norm.forward(xt)))[0, :, :, 0].numpy()
    return (pred_k - 273.15).astype(np.float32)


def clamp_layout(case, layout) -> Layout:
    """Shift each chiplet by the minimum amount to keep its bbox within the die outline."""
    from thermopt.layout.geometry import bounds as _bounds
    new_placements = []
    for p in layout.placements:
        x0, y0, x1, y1 = _bounds(case, p)
        dx = max(0.0, -x0) - max(0.0, x1 - case.outline_width)
        dy = max(0.0, -y0) - max(0.0, y1 - case.outline_height)
        new_placements.append(Placement(chiplet_id=p.chiplet_id,
                                        x=p.x + dx, y=p.y + dy,
                                        rotation=p.rotation))
    return Layout(placements=new_placements)


def hotspot_predict(case, layout, cache_path=None) -> np.ndarray:
    if cache_path and cache_path.exists():
        return np.load(cache_path)["hs"]
    layout = clamp_layout(case, layout)
    hs_backend = HotSpotBackend(case=case, config={"hotspot_binary": str(HS_BIN)})
    grid = hs_backend.simulate(case, layout)
    if cache_path:
        np.savez(cache_path, hs=grid)
    return grid


def load_milp_layout(case_name) -> Layout:
    d = json.load(open(MILP_DIR / case_name / "layout.json"))
    return Layout(placements=[
        Placement(chiplet_id=c["name"],
                  x=c.get("cx_mm", c["x"] * SCALE), y=c.get("cy_mm", c["y"] * SCALE),
                  rotation=int(round(math.degrees(c["angle_rad"]))) % 360)
        for c in d["chiplets"]
    ])


def load_opt_layout(run_dir, case_name) -> Layout:
    d = json.load(open(run_dir / case_name / "tmax/summary.json"))
    return Layout(placements=[
        Placement(chiplet_id=c["name"], x=c.get("cx_mm", c["x_mm"]), y=c.get("cy_mm", c["y_mm"]), rotation=c["rotation"])
        for c in d["chiplets"]
    ])


def metrics(grid):
    flat = np.sort(grid.flatten())[::-1]
    return float(flat[0]), float(flat[:50].mean())


def main(run_dir: Path):
    print(f"\nRun dir: {run_dir}")
    out_file = run_dir / "opt_comparison.png"

    fig, axes = plt.subplots(len(CASES), 6,
                             figsize=(24, 4.2 * len(CASES)),
                             constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.93, bottom=0.04,
                        hspace=0.45, wspace=0.35)

    col_titles = [
        "Initial — HotSpot", "Initial — UFNO", "Initial — Error (HS−UFNO)",
        "Optimized — HotSpot", "Optimized — UFNO", "Optimized — Error (HS−UFNO)",
    ]
    for j, t in enumerate(col_titles):
        axes[0, j].set_title(t, fontsize=10, pad=6)

    summary_rows = []

    for i, case_name in enumerate(CASES):
        opt_summary_path = run_dir / case_name / "tmax/summary.json"
        if not opt_summary_path.exists():
            print(f"  [SKIP] {case_name}: no optimization result")
            for ax in axes[i]: ax.axis("off")
            continue

        print(f"\n[{i+1}/{len(CASES)}] {case_name}", flush=True)
        ci   = load_atplace_case(CASES_DIR / case_name, {}, 42)
        case = ci.case
        W, H = case.outline_width, case.outline_height
        extent = [0, W, 0, H]

        layout_i = load_milp_layout(case_name)
        layout_o = load_opt_layout(run_dir, case_name)

        # --- UFNO predictions ---
        ufno_i = ufno_predict(case, layout_i)
        ufno_o = ufno_predict(case, layout_o)
        print(f"  UFNO  init={ufno_i.max():.1f}°C  opt={ufno_o.max():.1f}°C  Δ={ufno_o.max()-ufno_i.max():+.1f}°C")

        # --- HotSpot predictions (cached in run directory, same backend for consistency) ---
        cache_dir = run_dir / case_name
        cache_dir.mkdir(exist_ok=True)

        t0_hs = time.time()
        hs_i = hotspot_predict(case, layout_i, cache_dir / "hs_initial.npz")
        print(f"  HotSpot initial: {hs_i.max():.1f}°C ({time.time()-t0_hs:.0f}s)")

        t0_hs = time.time()
        hs_o = hotspot_predict(case, layout_o, cache_dir / "hs_optimized.npz")
        print(f"  HotSpot opt:     {hs_o.max():.1f}°C ({time.time()-t0_hs:.0f}s)")

        # Resample UFNO (64,64) to each HotSpot grid independently
        def resample_ufno(ufno_grid, hs_grid):
            zy = hs_grid.shape[0] / ufno_grid.shape[0]
            zx = hs_grid.shape[1] / ufno_grid.shape[1]
            return zoom(ufno_grid, (zy, zx), order=1)

        ufno_i_r = resample_ufno(ufno_i, hs_i)
        ufno_o_r = resample_ufno(ufno_o, hs_o)

        err_i = hs_i - ufno_i_r
        err_o = hs_o - ufno_o_r

        hs_tmax_i,  hs_t50_i  = metrics(hs_i)
        hs_tmax_o,  hs_t50_o  = metrics(hs_o)
        ufno_tmax_i, _        = metrics(ufno_i)
        ufno_tmax_o, _        = metrics(ufno_o)

        summary_rows.append({
            "case": case_name,
            "hs_tmax_i":   hs_tmax_i,
            "hs_tmax_o":   hs_tmax_o,
            "dhs_tmax":    hs_tmax_o - hs_tmax_i,
            "ufno_tmax_i": ufno_tmax_i,
            "ufno_tmax_o": ufno_tmax_o,
            "dufno_tmax":  ufno_tmax_o - ufno_tmax_i,
            "ufno_err_i":  ufno_tmax_i - hs_tmax_i,
            "ufno_err_o":  ufno_tmax_o - hs_tmax_o,
        })

        # --- Plot ---
        vmin = min(hs_i.min(), hs_o.min(), ufno_i_r.min(), ufno_o_r.min())
        vmax = max(hs_i.max(), hs_o.max(), ufno_i_r.max(), ufno_o_r.max())
        eabs = max(abs(err_i).max(), abs(err_o).max(), 1.0)

        arrays = [hs_i, ufno_i_r, err_i, hs_o, ufno_o_r, err_o]
        cmaps  = ["hot", "hot", "RdBu_r", "hot", "hot", "RdBu_r"]
        vmins  = [vmin]*2 + [-eabs] + [vmin]*2 + [-eabs]
        vmaxs  = [vmax]*2 + [eabs]  + [vmax]*2 + [eabs]

        for j, (arr, cmap, mn, mx) in enumerate(zip(arrays, cmaps, vmins, vmaxs)):
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
                ax.text(-0.36, 0.5, f"{case_name}\n({W:.0f}×{H:.0f}mm)",
                        transform=ax.transAxes, fontsize=9, fontweight="bold",
                        va="center", ha="center", rotation=90)
            # Annotate tmax on HS panels
            if j in (0, 3):
                tmax_val = hs_tmax_i if j == 0 else hs_tmax_o
                ax.set_title(f"HS Tmax={tmax_val:.1f}°C", fontsize=7, pad=2, color="#333")
            if j in (1, 4):
                tmax_val = ufno_tmax_i if j == 1 else ufno_tmax_o
                hs_ref   = hs_tmax_i  if j == 1 else hs_tmax_o
                ax.set_title(f"UFNO Tmax={tmax_val:.1f}°C (err={tmax_val-hs_ref:+.1f}°C)",
                             fontsize=7, pad=2, color="#333")

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.04)
            cb  = fig.colorbar(im, cax=cax)
            cb.ax.tick_params(labelsize=6)
            cb.set_label("°C (diff)" if j in (2, 5) else "°C", fontsize=6)

    fig.suptitle("UFNO Thermal Optimization: Before vs After (HotSpot ground truth + UFNO prediction)",
                 fontsize=12, y=0.97)
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out_file}")

    # Summary table
    print(f"\n{'='*100}")
    print(f"{'Case':7} | {'HS_Tmax_i':>10} {'HS_Tmax_o':>10} {'ΔHS_Tmax':>9} | "
          f"{'UFNO_i':>7} {'UFNO_o':>7} {'ΔUFNO':>7} | "
          f"{'UFNO_err_i':>11} {'UFNO_err_o':>11}")
    print("-" * 100)
    for r in summary_rows:
        print(f"{r['case']:7} | {r['hs_tmax_i']:>10.2f} {r['hs_tmax_o']:>10.2f} {r['dhs_tmax']:>+9.2f} | "
              f"{r['ufno_tmax_i']:>7.2f} {r['ufno_tmax_o']:>7.2f} {r['dufno_tmax']:>+7.2f} | "
              f"{r['ufno_err_i']:>+11.2f} {r['ufno_err_o']:>+11.2f}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, help="Path to optimization run directory")
    args = parser.parse_args()
    main(Path(args.run_dir))
