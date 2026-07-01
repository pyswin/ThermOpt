"""
spacing_repair_eval.py — 对现有布局做最小间距修复，然后跑热仿真对比

流程：
  1. 读入 ATPlace/Gurobi 跑出来的 milp150s 布局
  2. 对间距 < min_spacing 的 chiplet 对进行迭代推开
  3. 用 HotSpot 和 UFNO 分别测温，输出对比表

用法:
    python3 scripts/spacing_repair_eval.py --spacing 0.2 --cases Case3 Case5 Case6 Case7 Case8
"""
import argparse, json, math, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/thermopt/thermal/ufno_demo"))

from thermopt.data.atplace import load_atplace_case
from thermopt.layout.objects import Layout, Placement
from thermopt.layout.geometry import bounds as chiplet_bounds
from thermopt.thermal.surrogate_input import (
    coordinate_grid, rasterize_power_channel, SURROGATE_NATIVE_GRID_SIZE,
)
from thermopt.thermal.hotspot import HotSpotBackend

CASES_DIR  = ROOT / "external/ATPlace_pub/cases"
MILP150S   = ROOT / "atplace/20260627_163106_milp150s"
UFNO_PT    = ROOT / "src/thermopt/thermal/ufno_demo/model.pt"
HS_BIN     = ROOT / "external/ATPlace_pub/thermal/hotspot"
SCALE      = 0.001

_UFNO = {}
def get_ufno():
    if not _UFNO:
        xn, m, yn = torch.load(str(UFNO_PT), map_location="cpu", weights_only=False)
        m.eval()
        _UFNO["s"] = (xn, m, yn)
        print(f"[UFNO] loaded ({sum(p.numel() for p in m.parameters())/1e6:.1f}M params)")
    return _UFNO["s"]


def load_milp150s_layout(case_name: str) -> Layout:
    d = json.load(open(MILP150S / case_name / "layout.json"))
    return Layout(placements=[
        Placement(chiplet_id=c["name"],
                  x=c["x"] * SCALE, y=c["y"] * SCALE,
                  rotation=int(round(math.degrees(c["angle_rad"]))) % 360)
        for c in d["chiplets"]
    ])


def repair_spacing(case, layout: Layout, min_spacing: float, max_iter: int = 200) -> tuple[Layout, int, float]:
    """迭代推开间距不足的 chiplet 对，返回 (修复后布局, 迭代次数, 最大实际间距违规)"""
    placements = list(layout.placements)
    n = len(placements)

    def get_box(p):
        x0, y0, x1, y1 = chiplet_bounds(case, p)
        return x0, y0, x1, y1

    for iteration in range(max_iter):
        max_violation = 0.0
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                pi, pj = placements[i], placements[j]
                xi0, yi0, xi1, yi1 = get_box(pi)
                xj0, yj0, xj1, yj1 = get_box(pj)

                # 每个方向的间距（正值=有间隙，负值=重叠）
                gap_left  = xj0 - xi1   # i 在 j 左边
                gap_right = xi0 - xj1   # i 在 j 右边
                gap_below = yj0 - yi1   # i 在 j 下边
                gap_above = yi0 - yj1   # i 在 j 上边

                # 主分离方向：四个里面最大的那个
                best_gap = max(gap_left, gap_right, gap_below, gap_above)

                if best_gap >= min_spacing:
                    continue   # 已满足

                max_violation = max(max_violation, min_spacing - best_gap)
                moved = True

                # 在主分离方向上推开
                need = (min_spacing - best_gap) / 2.0   # 每侧各推一半
                if best_gap == gap_left:
                    placements[i] = Placement(pi.chiplet_id, pi.x - need, pi.y, pi.rotation)
                    placements[j] = Placement(pj.chiplet_id, pj.x + need, pj.y, pj.rotation)
                elif best_gap == gap_right:
                    placements[i] = Placement(pi.chiplet_id, pi.x + need, pi.y, pi.rotation)
                    placements[j] = Placement(pj.chiplet_id, pj.x - need, pj.y, pj.rotation)
                elif best_gap == gap_below:
                    placements[i] = Placement(pi.chiplet_id, pi.x, pi.y - need, pi.rotation)
                    placements[j] = Placement(pj.chiplet_id, pj.x, pj.y + need, pj.rotation)
                else:
                    placements[i] = Placement(pi.chiplet_id, pi.x, pi.y + need, pi.rotation)
                    placements[j] = Placement(pj.chiplet_id, pj.x, pj.y - need, pj.rotation)

        if not moved:
            break

    return Layout(placements=placements), iteration + 1, max_violation


def count_violations(case, layout: Layout, min_spacing: float) -> tuple[int, float]:
    placements = list(layout.placements)
    n = len(placements)
    count = 0
    max_viol = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            x0i, y0i, x1i, y1i = chiplet_bounds(case, placements[i])
            x0j, y0j, x1j, y1j = chiplet_bounds(case, placements[j])
            gap = max(x0j - x1i, x0i - x1j, y0j - y1i, y0i - y1j)
            if gap < min_spacing:
                count += 1
                max_viol = max(max_viol, min_spacing - gap)
    return count, max_viol


def ufno_eval(case, layout):
    xn, m, yn = get_ufno()
    gx, gy = coordinate_grid(case, SURROGATE_NATIVE_GRID_SIZE)
    power = np.zeros((64, 64), dtype=np.float32)
    rasterize_power_channel(case, layout, gx, gy, out=power)
    x = np.stack([power, gx, gy], axis=-1)[:, :, np.newaxis, :]
    with torch.no_grad():
        pred_k = yn.inverse(m(xn.forward(torch.from_numpy(x).unsqueeze(0))))[0, :, :, 0].numpy()
    temp = pred_k - 273.15
    flat = np.sort(temp.flatten())[::-1]
    return float(flat[0]), float(flat[:50].mean())


def hotspot_eval(case, layout):
    hs = HotSpotBackend(case=case, config={"hotspot_binary": str(HS_BIN)})
    grid = hs.simulate(case, layout)
    flat = np.sort(grid.flatten())[::-1]
    return float(flat[0]), float(flat[:50].mean())


def main(min_spacing: float, cases: list[str]):
    get_ufno()
    print(f"\nmin_spacing = {min_spacing:.2f}mm  |  cases: {cases}")
    print(f"\n{'Case':7} | {'violations':>10}  {'iters':>5} | {'HS_Tmax':>8}  {'UFNO_Tmax':>10}  {'err':>7} | {'HSt':>5}")
    print("-" * 75)

    results = {}
    for cn in cases:
        ci = load_atplace_case(CASES_DIR / cn, {}, 42)
        case = ci.case

        layout_orig = load_milp150s_layout(cn)
        viol_before, _ = count_violations(case, layout_orig, min_spacing)

        layout_fixed, iters, _ = repair_spacing(case, layout_orig, min_spacing)
        viol_after, max_viol = count_violations(case, layout_fixed, min_spacing)

        u_tmax, u_t50 = ufno_eval(case, layout_fixed)

        t0 = time.time()
        hs_tmax, hs_t50 = hotspot_eval(case, layout_fixed)
        hst = time.time() - t0

        err = u_tmax - hs_tmax
        print(f"{cn:7} | {viol_before:>4}→{viol_after:<4}  {iters:>5} | "
              f"{hs_tmax:>8.2f}  {u_tmax:>10.2f}  {err:>+7.2f} | {hst:>4.0f}s")
        results[cn] = dict(hs_tmax=hs_tmax, hs_t50=hs_t50, ufno_tmax=u_tmax,
                           ufno_t50=u_t50, err_tmax=err, violations_before=viol_before)

    errs = [abs(v["err_tmax"]) for v in results.values()]
    print(f"\nMean |err_Tmax| = {np.mean(errs):.2f}°C")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--spacing", type=float, default=0.2)
    parser.add_argument("--cases", nargs="+", default=["Case3","Case5","Case6","Case7","Case8"])
    args = parser.parse_args()
    main(args.spacing, args.cases)
