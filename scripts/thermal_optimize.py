"""
thermal_optimize.py — 热感知布局优化主脚本（UFNO 热模型，无 WL 约束）

默认有效 case：Case3, Case5, Case6, Case7, Case8
热目标：tmax 和 tmax50 各跑一次
结果保存至 atplace/thermal_runs/{timestamp}_{tag}/
"""
import sys, json, math, time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/thermopt/thermal/ufno_demo"))

import numpy as np
import torch

from thermopt.data.atplace import load_atplace_case
from thermopt.layout.objects import Layout, Placement
from thermopt.layout.geometry import hpwl
from thermopt.optimizer.atplace import _analytical_refine
from thermopt.thermal.surrogate_input import (
    coordinate_grid, rasterize_power_channel, SURROGATE_NATIVE_GRID_SIZE,
)

SCALE     = 0.001
CASES_DIR = ROOT / "external/ATPlace_pub/cases"
MILP_DIR  = ROOT / "atplace/20260627_163106_milp150s"
UFNO_PT   = ROOT / "src/thermopt/thermal/ufno_demo/model.pt"
RUNS_DIR  = ROOT / "atplace/thermal_runs"

VALID_CASES = ["Case3", "Case5", "Case6", "Case7", "Case8"]

BASE_CONFIG = dict(
    refine_steps=800,
    learning_rate=0.03,
    density_weight=5000.0,
    outline_weight=20000.0,
    wl_weight=1.0,
    wl_budget_weight=0.0,
    wl_budget_factor=0.0,
)

# UFNO model cache
_UFNO_CACHE: dict = {}

def _get_ufno():
    if "state" not in _UFNO_CACHE:
        x_norm, model, y_norm = torch.load(str(UFNO_PT), map_location="cpu", weights_only=False)
        model.eval()
        _UFNO_CACHE["state"] = (x_norm, model, y_norm)
        print(f"[UFNO eval] loaded ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")
    return _UFNO_CACHE["state"]


def load_layout(path: Path) -> list[Placement]:
    with open(path) as f:
        d = json.load(f)
    return [
        Placement(
            chiplet_id=c["name"],
            x=c["x"] * SCALE,
            y=c["y"] * SCALE,
            rotation=int(round(math.degrees(c["angle_rad"]))) % 360,
        )
        for c in d["chiplets"]
    ]


def layout_to_dict(case, layout):
    return [
        {
            "name": p.chiplet_id,
            "x_mm": round(p.x, 6),
            "y_mm": round(p.y, 6),
            "rotation": p.rotation,
        }
        for p in layout.placements
    ]


def ufno_eval(case, layout):
    """Evaluate layout with UFNO: returns (tmax_c, t50_c, wl_m)."""
    x_norm, model, y_norm = _get_ufno()
    gx, gy = coordinate_grid(case, SURROGATE_NATIVE_GRID_SIZE)
    power  = np.zeros((64, 64), dtype=np.float32)
    rasterize_power_channel(case, layout, gx, gy, out=power)
    x_phys = np.stack([power, gx, gy], axis=-1)[:, :, np.newaxis, :]
    xt = torch.from_numpy(x_phys).unsqueeze(0)
    with torch.no_grad():
        pred_k = y_norm.inverse(model(x_norm.forward(xt)))[0, :, :, 0].numpy()
    temp_c = pred_k - 273.15
    flat   = np.sort(temp_c.flatten())[::-1]
    wl_m   = hpwl(case, layout) / 1e3
    return float(flat[0]), float(flat[:50].mean()), wl_m


def main(cases=None, thermal_weight=10000, tag=""):
    if cases is None:
        cases = VALID_CASES
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = f"{stamp}_{tag}" if tag else f"{stamp}_allcases_tw{thermal_weight}"
    run_dir = RUNS_DIR / label
    run_dir.mkdir(parents=True, exist_ok=True)

    _get_ufno()   # preload model
    print(f"[{label}]  tw={thermal_weight}  cases={cases}\n")

    all_results = {}
    t_wall_start = time.time()

    modes = [("tmax", "tmax")]

    for case_name in cases:
        print(f"{'='*60}")
        print(f"  {case_name}")
        print(f"{'='*60}")

        case_input = load_atplace_case(CASES_DIR / case_name, {}, 42)
        case = case_input.case

        init_placements = load_layout(MILP_DIR / case_name / "layout.json")
        layout0 = Layout(placements=init_placements)

        t0, t50_0, wl0 = ufno_eval(case, layout0)
        atp_wl = json.load(open(MILP_DIR / case_name / "summary.json"))["twl_m"]

        print(f"  Initial:  HPWL={wl0:.3f}m (ATPlace TWL={atp_wl:.3f}m)  Tmax={t0:.1f}C  Tmax50={t50_0:.1f}C\n")

        case_results = {
            "initial": {"thermopt_hpwl_m": wl0, "atplace_twl_m": atp_wl,
                        "tmax_c": t0, "tmax50_c": t50_0}
        }
        case_dir = run_dir / case_name
        case_dir.mkdir(exist_ok=True)

        for m_key, m_mode in modes:
            config = {
                **BASE_CONFIG,
                "thermal_weight":     float(thermal_weight),
                "thermal_mode":       m_mode,
                "thermal_model_path": str(UFNO_PT),
            }

            t_start = time.time()
            layout1 = _analytical_refine(case, layout0, config, seed=42)
            elapsed = time.time() - t_start

            t1, t50_1, wl1 = ufno_eval(case, layout1)
            dwl   = (wl1 - wl0) / wl0 * 100
            dt    = t1 - t0
            dt50  = t50_1 - t50_0

            if m_mode == "tmax":
                print(f"  [Tmax  ]  WL {wl0:.3f}→{wl1:.3f}m ({dwl:+.1f}%)  "
                      f"Tmax {t0:.1f}→{t1:.1f}C ({dt:+.1f}C)  Tmax50={t50_1:.1f}C  {elapsed:.0f}s")
            else:
                print(f"  [Tmax50]  WL {wl0:.3f}→{wl1:.3f}m ({dwl:+.1f}%)  "
                      f"Tmax50 {t50_0:.1f}→{t50_1:.1f}C ({dt50:+.1f}C)  Tmax={t1:.1f}C  {elapsed:.0f}s")

            result = {
                "case": case_name, "run": m_key, "timestamp": stamp,
                "thermal_weight": thermal_weight, "thermal_mode": m_mode,
                "initial": {"thermopt_hpwl_m": wl0, "atplace_twl_m": atp_wl,
                            "tmax_c": t0, "tmax50_c": t50_0},
                "final":   {"wl_m": wl1, "tmax_c": t1, "tmax50_c": t50_1},
                "delta":   {"wl_pct": dwl, "dtmax_c": dt, "dtmax50_c": dt50},
                "runtime_s": elapsed,
                "config": config,
                "chiplets": layout_to_dict(case, layout1),
            }
            out_dir = case_dir / m_key
            out_dir.mkdir(exist_ok=True)
            with open(out_dir / "summary.json", "w") as f:
                json.dump(result, f, indent=2)
            case_results[m_key] = result

        print()
        all_results[case_name] = case_results

    total = time.time() - t_wall_start
    with open(run_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary table
    print(f"\n{'='*90}")
    print(f"SUMMARY  [{label}]  total: {total/60:.1f} min")
    print(f"{'='*90}")
    hdr = (f"{'Case':7} {'Mode':7} | {'WLinit':>8} {'WLfinal':>9} {'ΔWL%':>6} | "
           f"{'Tmax_i':>7} {'Tmax_f':>7} {'ΔTmax':>7} | "
           f"{'T50_i':>6} {'T50_f':>7} {'ΔTmax50':>8} | {'Time':>5}")
    print(hdr)
    print('-' * len(hdr))
    for case_name in cases:
        cr = all_results[case_name]
        init = cr["initial"]
        for m_key, _ in modes:
            if m_key not in cr:
                continue
            r  = cr[m_key]
            d  = r["delta"]
            f_ = r["final"]
            print(f"{case_name:7} {m_key:7} | "
                  f"{init['thermopt_hpwl_m']:>8.3f} {f_['wl_m']:>9.3f} {d['wl_pct']:>+6.1f}% | "
                  f"{init['tmax_c']:>7.1f} {f_['tmax_c']:>7.1f} {d['dtmax_c']:>+7.1f}C | "
                  f"{init['tmax50_c']:>6.1f} {f_['tmax50_c']:>7.1f} {d['dtmax50_c']:>+8.1f}C | "
                  f"{r['runtime_s']:>5.0f}s")
        print()

    print(f"Saved to: {run_dir}")
    return run_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=VALID_CASES)
    parser.add_argument("--weight", type=int, default=10000)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    main(cases=args.cases, thermal_weight=args.weight, tag=args.tag)
