"""
Run HotSpot ground-truth simulation on initial (MILP) and optimized layouts
from the all-cases maxT run, then print the corrected comparison table.
"""
import sys, json, math, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from thermopt.data.atplace import load_atplace_case
from thermopt.layout.objects import Layout, Placement
from thermopt.thermal.hotspot import HotSpotBackend

SCALE     = 0.001
CASES_DIR = Path(__file__).parent.parent / "external/ATPlace_pub/cases"
MILP_DIR  = Path(__file__).parent.parent / "atplace/20260627_163106_milp150s"
HOTSPOT   = str(Path(__file__).parent.parent / "external/ATPlace_pub/thermal/hotspot")
HOTSPOT_CFG = {"hotspot_binary": HOTSPOT}

ALLCASES_RUN = Path(__file__).parent.parent / "atplace/thermal_runs/20260628_004144_maxT"
VALID_CASES  = ["Case3", "Case5", "Case6", "Case7", "Case8"]
MODES        = ["tmax", "tmax50"]


def load_layout_from_milp(case_name: str) -> Layout:
    d = json.load(open(MILP_DIR / case_name / "layout.json"))
    return Layout(placements=[
        Placement(chiplet_id=c["name"],
                  x=c["x"] * SCALE, y=c["y"] * SCALE,
                  rotation=int(round(math.degrees(c["angle_rad"]))) % 360)
        for c in d["chiplets"]
    ])


def load_layout_from_summary(summary_path: Path) -> Layout:
    d = json.load(open(summary_path))
    return Layout(placements=[
        Placement(chiplet_id=c["name"],
                  x=c["x_mm"], y=c["y_mm"],
                  rotation=c["rotation"])
        for c in d["chiplets"]
    ])


def hotspot_temps(backend, case, layout):
    temp = backend.simulate(case, layout)
    flat = sorted(temp.flatten(), reverse=True)
    return float(flat[0]), float(sum(flat[:50]) / 50)


def hotspot_temps_safe(backend, case, layout):
    """Return (tmax, tmax50) or None if HotSpot fails."""
    try:
        return hotspot_temps(backend, case, layout)
    except Exception as e:
        print(f"    [WARN] HotSpot failed: {e}")
        return None


def main():
    # Load partial results from a previous run if available
    out = ALLCASES_RUN / "hotspot_validation.json"
    results = json.load(open(out)) if out.exists() else {}

    for case_name in VALID_CASES:
        print(f"\n{'='*56}\n  {case_name}\n{'='*56}")
        ci = load_atplace_case(CASES_DIR / case_name, {}, 42)
        case = ci.case
        backend = HotSpotBackend(case=case, config=HOTSPOT_CFG)

        wl0 = json.load(open(ALLCASES_RUN / case_name / "tmax" / "summary.json"))["initial"]["thermopt_hpwl_m"]

        # Initial layout — reuse if already done
        if case_name in results and "initial" in results[case_name]:
            r0 = results[case_name]["initial"]
            tmax0, t50_0 = r0["tmax_c"], r0["tmax50_c"]
            print(f"  Initial : Tmax={tmax0:.2f}C  Tmax50={t50_0:.2f}C  (cached)")
        else:
            t0 = time.time()
            layout0 = load_layout_from_milp(case_name)
            res0 = hotspot_temps_safe(backend, case, layout0)
            if res0 is None:
                print(f"  Initial : FAILED — skipping case")
                continue
            tmax0, t50_0 = res0
            print(f"  Initial : Tmax={tmax0:.2f}C  Tmax50={t50_0:.2f}C  ({time.time()-t0:.0f}s)")

        case_res = results.get(case_name, {})
        case_res["initial"] = {"tmax_c": tmax0, "tmax50_c": t50_0, "wl_m": wl0}

        for mode in MODES:
            if mode in case_res and "tmax_c" in case_res[mode]:
                r = case_res[mode]
                print(f"  [{mode:6}]: Tmax={r['tmax_c']:.2f}C  Tmax50={r['tmax50_c']:.2f}C  (cached)")
                continue

            summ_path = ALLCASES_RUN / case_name / mode / "summary.json"
            summ = json.load(open(summ_path))
            layout1 = load_layout_from_summary(summ_path)
            wl1 = summ["final"]["wl_m"]
            t0 = time.time()
            res1 = hotspot_temps_safe(backend, case, layout1)
            scot_tmax = summ["final"]["tmax_c"]
            scot_t50  = summ["final"]["tmax50_c"]
            if res1 is None:
                print(f"  [{mode:6}]: FAILED (ScOT: {scot_tmax:.1f}/{scot_t50:.1f})")
                case_res[mode] = {
                    "wl_m": wl1, "tmax_c": None, "tmax50_c": None,
                    "scot_tmax_c": scot_tmax, "scot_t50_c": scot_t50,
                    "dtmax_c": None, "dtmax50_c": None,
                    "wl_pct": (wl1 - wl0) / wl0 * 100, "failed": True,
                }
            else:
                tmax1, t50_1 = res1
                print(f"  [{mode:6}]: Tmax={tmax1:.2f}C  Tmax50={t50_1:.2f}C  "
                      f"(ScOT: {scot_tmax:.1f}/{scot_t50:.1f})  ({time.time()-t0:.0f}s)")
                case_res[mode] = {
                    "wl_m": wl1,
                    "tmax_c": tmax1, "tmax50_c": t50_1,
                    "scot_tmax_c": scot_tmax, "scot_t50_c": scot_t50,
                    "dtmax_c": tmax1 - tmax0, "dtmax50_c": t50_1 - t50_0,
                    "wl_pct": (wl1 - wl0) / wl0 * 100,
                }

        results[case_name] = case_res
        json.dump(results, open(out, "w"), indent=2)

    print(f"\nSaved to: {out}")

    # Summary table
    print(f"\n{'='*96}")
    print("HOTSPOT-VALIDATED RESULTS  [tw=10000, no WL constraint]")
    print(f"{'='*96}")
    hdr = (f"{'Case':7} {'Mode':7} | "
           f"{'WLinit':>8} {'WLfinal':>9} {'ΔWL%':>6} | "
           f"{'Tmax_i':>8} {'Tmax_f':>8} {'ΔTmax':>7} | "
           f"{'T50_i':>7} {'T50_f':>8} {'ΔTmax50':>8} | "
           f"{'ScOT_err':>9}")
    print(hdr)
    print('-' * len(hdr))
    def _fmt(v, spec):
        return (spec % v) if v is not None else "N/A".rjust(len(spec % 0))

    for case_name in VALID_CASES:
        if case_name not in results:
            continue
        cr = results[case_name]
        init = cr["initial"]
        for mode in MODES:
            if mode not in cr:
                continue
            r     = cr[mode]
            t_f   = r.get("tmax_c")
            t50_f = r.get("tmax50_c")
            scot  = r.get("scot_tmax_c")
            err   = (scot - t_f) if (scot is not None and t_f is not None) else None
            tag   = " [*]" if r.get("failed") else ""
            print(f"{case_name:7} {mode:7} | "
                  f"{init['wl_m']:>8.3f} {_fmt(r.get('wl_m'),'%9.3f')} {_fmt(r.get('wl_pct'),'%+6.1f')}% | "
                  f"{init['tmax_c']:>8.2f} {_fmt(t_f,'%8.2f')} {_fmt(r.get('dtmax_c'),'%+7.2f')}C | "
                  f"{init['tmax50_c']:>7.2f} {_fmt(t50_f,'%8.2f')} {_fmt(r.get('dtmax50_c'),'%+8.2f')}C | "
                  f"{_fmt(scot,'%9.1f')} {_fmt(err,'%+7.2f')}C{tag}")
        print()
    print("[*] HotSpot failed (chiplet out-of-bounds by <0.1mm, soft-constraint artifact)")


if __name__ == "__main__":
    main()
