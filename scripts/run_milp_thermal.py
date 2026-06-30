"""
run_milp_thermal.py — MILP布局 + 热仿真精度对比

对每个 case 跑 MILP（可配置间距和时间），用 UFNO 和 HotSpot 分别预测温度，
输出精度对比表。

用法:
    python3 scripts/run_milp_thermal.py                        # 默认 0mm spacing
    python3 scripts/run_milp_thermal.py --spacing 0.05         # 0.05mm spacing
    python3 scripts/run_milp_thermal.py --spacing 0.05 --time 150
"""
import argparse, json, math, sys, time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/thermopt/thermal/ufno_demo"))

from thermopt.data.atplace import load_atplace_case
from thermopt.layout.objects import Layout, Placement
from thermopt.layout.geometry import hpwl
from thermopt.optimizer.atplace import _solve_clump_milp
from thermopt.thermal.surrogate_input import (
    coordinate_grid, rasterize_power_channel, SURROGATE_NATIVE_GRID_SIZE,
)
from thermopt.thermal.hotspot import HotSpotBackend

CASES_DIR = ROOT / "external/ATPlace_pub/cases"
UFNO_PT   = ROOT / "src/thermopt/thermal/ufno_demo/model.pt"
HS_BIN    = ROOT / "external/ATPlace_pub/thermal/hotspot"
OUT_ROOT  = ROOT / "atplace/milp_spacing_runs"
VALID_CASES = ["Case3", "Case5", "Case6", "Case7", "Case8"]

_UFNO = {}
def get_ufno():
    if not _UFNO:
        xn, m, yn = torch.load(str(UFNO_PT), map_location="cpu", weights_only=False)
        m.eval()
        _UFNO["s"] = (xn, m, yn)
        print(f"[UFNO] loaded ({sum(p.numel() for p in m.parameters())/1e6:.1f}M params)")
    return _UFNO["s"]


def ufno_eval(case, layout):
    x_norm, model, y_norm = get_ufno()
    gx, gy = coordinate_grid(case, SURROGATE_NATIVE_GRID_SIZE)
    power  = np.zeros((64, 64), dtype=np.float32)
    rasterize_power_channel(case, layout, gx, gy, out=power)
    x_phys = np.stack([power, gx, gy], axis=-1)[:, :, np.newaxis, :]
    xt = torch.from_numpy(x_phys).unsqueeze(0)
    with torch.no_grad():
        pred_k = y_norm.inverse(model(x_norm.forward(xt)))[0, :, :, 0].numpy()
    temp = pred_k - 273.15
    flat = np.sort(temp.flatten())[::-1]
    return float(flat[0]), float(flat[:50].mean())


def hotspot_eval(case, layout):
    hs = HotSpotBackend(case=case, config={"hotspot_binary": str(HS_BIN)})
    grid = hs.simulate(case, layout)
    flat = np.sort(grid.flatten())[::-1]
    return float(flat[0]), float(flat[:50].mean()), grid


def main(spacing: float = 0.0, time_limit: float = 150.0, cases=None):
    if cases is None:
        cases = VALID_CASES

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag   = f"sp{spacing:.3f}mm_t{int(time_limit)}s"
    run_dir = OUT_ROOT / f"{stamp}_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "milp_time_limit":    time_limit,
        "mip_rel_gap":        0.01,
        "min_chiplet_spacing": spacing,
        "verbose":            False,
    }

    get_ufno()  # preload

    results = {}
    t_wall = time.time()

    hdr = (f"\n{'Case':7} | {'WL(m)':>8} {'MILPt':>6} | "
           f"{'HS_Tmax':>8} {'HS_T50':>7} | "
           f"{'UFNO_Tmax':>10} {'err_Tmax':>9} | "
           f"{'HSt':>5}")
    print(hdr)
    print("-" * len(hdr))

    for case_name in cases:
        ci   = load_atplace_case(CASES_DIR / case_name, {}, 42)
        case = ci.case

        # --- MILP ---
        t0 = time.time()
        layout, ok, msg, obj = _solve_clump_milp(case, config)
        milp_t = time.time() - t0
        wl_m   = hpwl(case, layout) / 1e3

        # --- UFNO ---
        ufno_tmax, ufno_t50 = ufno_eval(case, layout)

        # --- HotSpot ---
        t0 = time.time()
        hs_tmax, hs_t50, hs_grid = hotspot_eval(case, layout)
        hs_t = time.time() - t0

        err_tmax = ufno_tmax - hs_tmax

        print(f"{case_name:7} | {wl_m:>8.3f} {milp_t:>5.0f}s | "
              f"{hs_tmax:>8.2f} {hs_t50:>7.2f} | "
              f"{ufno_tmax:>10.2f} {err_tmax:>+9.2f} | "
              f"{hs_t:>4.0f}s")

        case_dir = run_dir / case_name
        case_dir.mkdir()
        np.savez(case_dir / "hs_grid.npz", hs=hs_grid)

        # Save layout
        chiplets_out = [{"name": p.chiplet_id, "x_mm": round(p.x, 6),
                         "y_mm": round(p.y, 6), "rotation": p.rotation}
                        for p in layout.placements]
        results[case_name] = {
            "wl_m": wl_m, "milp_time_s": milp_t, "milp_success": ok,
            "hs_tmax": hs_tmax, "hs_t50": hs_t50,
            "ufno_tmax": ufno_tmax, "ufno_t50": ufno_t50,
            "err_tmax": err_tmax, "err_t50": ufno_t50 - hs_t50,
            "hotspot_time_s": hs_t,
            "chiplets": chiplets_out,
        }
        json.dump(results[case_name], open(case_dir / "summary.json", "w"), indent=2)

    total = time.time() - t_wall
    json.dump({"config": config, "cases": results},
              open(run_dir / "all_results.json", "w"), indent=2)

    # Summary
    print(f"\n{'='*70}")
    print(f"spacing={spacing}mm  time_limit={time_limit}s  total={total/60:.1f}min")
    print(f"{'Case':7} | {'|err_Tmax|':>10} {'|err_T50|':>9} | {'WL(m)':>8}")
    print("-" * 45)
    errs_tmax, errs_t50 = [], []
    for cn, r in results.items():
        print(f"{cn:7} | {r['err_tmax']:>+10.2f} {r['err_t50']:>+9.2f} | {r['wl_m']:>8.3f}")
        errs_tmax.append(abs(r["err_tmax"]))
        errs_t50.append(abs(r["err_t50"]))
    print(f"{'Mean':7} | {np.mean(errs_tmax):>10.2f} {np.mean(errs_t50):>9.2f}")
    print(f"\nSaved → {run_dir}")
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--spacing", type=float, default=0.05, help="min chiplet spacing mm")
    parser.add_argument("--time",    type=float, default=150.0, help="MILP time limit s")
    parser.add_argument("--cases",   nargs="+",  default=VALID_CASES)
    args = parser.parse_args()
    main(spacing=args.spacing, time_limit=args.time, cases=args.cases)
