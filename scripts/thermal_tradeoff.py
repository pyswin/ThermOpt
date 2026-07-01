"""
thermal_tradeoff.py — WL-Thermal 权衡分析脚本

对每组 (case, thermal_weight, WL预算, 热目标) 组合独立运行梯度优化，
输出系统性对比表，用于分析线长代价与降温幅度之间的权衡关系。

WL 预算分两档：
  tight — case 特定上限（TIGHT_BUDGET 字典，如 Case3: 1.42x, Case5: 1.08x）
  loose — 2.0x（允许线长最多翻倍）

用法：
  python3 scripts/thermal_tradeoff.py --cases Case3 Case5 --weights 2000 5000 10000
"""
import sys, json, math, time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from thermopt.data.atplace import load_atplace_case
from thermopt.layout.objects import Layout, Placement
from thermopt.layout.geometry import hpwl
from thermopt.optimizer.atplace import _analytical_refine
from thermopt.thermal.thermfm import ThermFMThermalBackend

SCALE     = 0.001
CASES_DIR = Path(__file__).parent.parent / "external/ATPlace_pub/cases"
MILP_DIR  = Path(__file__).parent.parent / "atplace/20260627_163106_milp150s"
MODEL_DIR = Path(__file__).parent.parent / "src/thermopt/thermal/thermfm_t_case_all_demo/model"
RUNS_DIR  = Path(__file__).parent.parent / "atplace/thermal_runs"

BASE_CONFIG = dict(
    refine_steps=300,
    learning_rate=0.05,
    density_weight=5000.0,
    outline_weight=20000.0,
    wl_weight=1.0,
    wl_budget_weight=1e5,
)

# Case-specific tight WL budget factors relative to our hpwl()
TIGHT_BUDGET = {
    "Case3": 1.42,
    "Case5": 1.08,
}


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
    rows = []
    for p in layout.placements:
        chip = case.chiplet_by_id[p.chiplet_id]
        rows.append({
            "name": p.chiplet_id,
            "x_mm": round(p.x, 6),
            "y_mm": round(p.y, 6),
            "rotation": p.rotation,
            "width_mm": chip.width,
            "height_mm": chip.height,
            "power_w": chip.power,
        })
    return rows


def eval_layout(case, layout, backend):
    wl_m = hpwl(case, layout) / 1e3
    temp_map = backend.simulate(case, layout)
    flat = sorted(temp_map.flatten(), reverse=True)
    tmax   = float(flat[0])
    tmax50 = float(sum(flat[:50]) / 50)
    return wl_m, tmax, tmax50


def load_atplace_wl(case_name: str) -> float:
    """Load ATPlace MILP TWL (m) for a case — the authoritative WL baseline."""
    summ = json.load(open(MILP_DIR / case_name / "summary.json"))
    return summ["twl_m"]


def main(cases, thermal_weights=(2000,), tag=""):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = f"{stamp}_{tag}" if tag else stamp
    run_dir = RUNS_DIR / label
    run_dir.mkdir(parents=True, exist_ok=True)

    from thermopt.thermal.grad_thermal import load_scot_for_grad
    load_scot_for_grad(str(MODEL_DIR))
    print(f"[{label}] ScOT loaded. Cases: {cases}  thermal_weights: {thermal_weights}\n")

    all_results = {}
    t_total_start = time.time()

    for case_name in cases:
        print(f"{'='*70}")
        print(f"  {case_name}")
        print(f"{'='*70}")

        case_input = load_atplace_case(CASES_DIR / case_name, {}, 42)
        case = case_input.case
        backend = ThermFMThermalBackend(
            case=case,
            config={"backend": "thermfm", "thermfm_model_dir": str(MODEL_DIR)},
        )

        init_placements = load_layout(MILP_DIR / case_name / "layout.json")
        layout0 = Layout(placements=init_placements)

        wl0, t0, t50_0 = eval_layout(case, layout0, backend)
        atp_wl = load_atplace_wl(case_name)   # ATPlace authoritative WL
        tight_factor = TIGHT_BUDGET.get(case_name, 1.20)

        print(f"  MILP baseline: ATPlace TWL={atp_wl:.3f}m | ThermOpt HPWL={wl0:.3f}m")
        print(f"  Thermal init:  Tmax={t0:.1f}C  Tmax50={t50_0:.1f}C\n")

        groups = [
            ("tight", tight_factor),
            ("loose", 2.0),
        ]
        modes = [
            ("tmax",   "tmax"),
            ("tmax50", "tmax50"),
        ]

        case_results = {
            "milp_baseline": {"atplace_twl_m": atp_wl, "thermopt_hpwl_m": wl0,
                              "tmax_c": t0, "tmax50_c": t50_0}
        }
        case_dir = run_dir / case_name
        case_dir.mkdir(exist_ok=True)

        for tw in thermal_weights:
            for g_name, wl_factor in groups:
                wl_limit = wl0 * wl_factor
                print(f"  -- tw={tw}  {g_name} budget (≤{wl_factor:.0%}={wl_limit:.2f}m) --")

                for m_key, m_mode in modes:
                    run_key = f"tw{tw}_{g_name}_{m_key}"
                    config = {
                        **BASE_CONFIG,
                        "thermal_weight": float(tw),
                        "thermal_mode":   m_mode,
                        "wl_budget_factor": wl_factor,
                        "thermfm_model_dir": str(MODEL_DIR),
                    }

                    t_start = time.time()
                    layout1 = _analytical_refine(case, layout0, config, seed=42)
                    elapsed = time.time() - t_start

                    wl1, t1, t50_1 = eval_layout(case, layout1, backend)

                    # WL change vs ATPlace baseline (authoritative)
                    dwl_vs_atp = (wl1 - atp_wl) / atp_wl * 100
                    # WL change vs our init hpwl (optimizer's reference)
                    dwl_vs_init = (wl1 - wl0) / wl0 * 100
                    dt    = t1 - t0
                    dt50  = t50_1 - t50_0

                    if m_mode == "tmax":
                        primary = f"Tmax {t0:.1f}→{t1:.1f}C ({dt:+.1f}C)"
                    else:
                        primary = f"Tmax50 {t50_0:.1f}→{t50_1:.1f}C ({dt50:+.1f}C)"

                    print(f"    [{m_mode:6}] WL {wl0:.3f}→{wl1:.3f}m "
                          f"(vs init:{dwl_vs_init:+.1f}%  vs ATPlace:{dwl_vs_atp:+.1f}%)  "
                          f"{primary}  {elapsed:.0f}s")

                    out_dir = case_dir / run_key
                    out_dir.mkdir(exist_ok=True)
                    result = {
                        "case":            case_name,
                        "run":             run_key,
                        "timestamp":       stamp,
                        "group":           g_name,
                        "thermal_weight":  tw,
                        "thermal_mode":    m_mode,
                        "wl_budget_factor": wl_factor,
                        "wl_budget_m":     wl_limit,
                        "milp_baseline":   {"atplace_twl_m": atp_wl, "thermopt_hpwl_m": wl0,
                                            "tmax_c": t0, "tmax50_c": t50_0},
                        "final":           {"wl_m": wl1, "tmax_c": t1, "tmax50_c": t50_1},
                        "delta_vs_init":   {"wl_pct": dwl_vs_init, "dtmax_c": dt, "dtmax50_c": dt50},
                        "delta_vs_atp":    {"wl_pct": dwl_vs_atp},
                        "runtime_s":       elapsed,
                        "config":          config,
                        "chiplets":        layout_to_dict(case, layout1),
                    }
                    with open(out_dir / "summary.json", "w") as f:
                        json.dump(result, f, indent=2)
                    case_results[run_key] = result

                print()

        all_results[case_name] = case_results

    with open(run_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    total_elapsed = time.time() - t_total_start

    # ---- Final table ----
    print(f"\n{'='*100}")
    print(f"RESULTS  [{label}]  total runtime: {total_elapsed/60:.1f} min")
    print(f"{'='*100}")
    hdr = (f"{'Case':6} {'tw':>6} {'Budget':>7} {'Mode':>7} | "
           f"{'ATPlace':>8} {'ThrmHPWL':>9} {'WLfinal':>8} "
           f"{'ΔvsATP':>7} {'ΔvsInit':>8} | "
           f"{'Tmax':>6} {'ΔTmax':>7} | {'Tmax50':>7} {'ΔTmax50':>8} | {'Time':>5}")
    print(hdr)
    print('-' * len(hdr))

    for case_name in cases:
        cr = all_results[case_name]
        bl = cr["milp_baseline"]
        for tw in thermal_weights:
            for g_name, _ in groups:
                for m_key, _ in modes:
                    rk = f"tw{tw}_{g_name}_{m_key}"
                    if rk not in cr:
                        continue
                    r  = cr[rk]
                    f_ = r["final"]
                    di = r["delta_vs_init"]
                    da = r["delta_vs_atp"]
                    print(f"{case_name:6} {tw:>6} {g_name:>7} {m_key:>7} | "
                          f"{bl['atplace_twl_m']:>8.3f} {bl['thermopt_hpwl_m']:>9.3f} {f_['wl_m']:>8.3f} "
                          f"{da['wl_pct']:>+7.1f}% {di['wl_pct']:>+7.1f}% | "
                          f"{f_['tmax_c']:>6.1f} {di['dtmax_c']:>+7.1f}C | "
                          f"{f_['tmax50_c']:>7.1f} {di['dtmax50_c']:>+8.1f}C | "
                          f"{r['runtime_s']:>5.0f}s")
        print()

    print(f"\nSaved to: {run_dir}")
    return run_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=["Case3", "Case5"])
    parser.add_argument("--weights", nargs="+", type=int, default=[5000, 10000])
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    main(args.cases, thermal_weights=args.weights, tag=args.tag)
