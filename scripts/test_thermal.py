"""
独立热仿真测试脚本 — 与梯度/优化无关。

用 UFNO 模型对指定 case 的 MILP 初始布局做热预测，打印 Tmax / T50 / Tmin。
可选：同时用 HotSpot 验证（需要 Linux x86-64 环境）。

用法:
    python3 scripts/test_thermal.py --cases Case3 Case5
    python3 scripts/test_thermal.py --cases Case3 --hotspot
    python3 scripts/test_thermal.py --cases Case3 --layout custom_layout.json
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
from thermopt.thermal.surrogate_input import (
    coordinate_grid, rasterize_power_channel, SURROGATE_NATIVE_GRID_SIZE,
)

CASES_DIR  = ROOT / "external/ATPlace_pub/cases"
MILP_DIR   = ROOT / "atplace/20260627_163106_milp150s"
UFNO_MODEL = ROOT / "src/thermopt/thermal/ufno_demo/model.pt"
HOTSPOT_BIN = ROOT / "external/ATPlace_pub/thermal/hotspot"
SCALE = 0.001


# ---------------------------------------------------------------------------
# UFNO 推理
# ---------------------------------------------------------------------------
_UFNO_CACHE = {}

def load_ufno():
    if "model" not in _UFNO_CACHE:
        x_norm, model, y_norm = torch.load(
            str(UFNO_MODEL), map_location="cpu", weights_only=False)
        model.eval()
        _UFNO_CACHE["model"] = (x_norm, model, y_norm)
        n = sum(p.numel() for p in model.parameters())
        print(f"[UFNO] loaded  ({n/1e6:.1f}M params)")
    return _UFNO_CACHE["model"]


def ufno_predict(case, layout) -> np.ndarray:
    """返回 (64,64) 温度图，单位 °C。"""
    x_norm, model, y_norm = load_ufno()
    gx, gy = coordinate_grid(case, SURROGATE_NATIVE_GRID_SIZE)
    power = np.zeros((64, 64), dtype=np.float32)
    rasterize_power_channel(case, layout, gx, gy, out=power)
    # UFNO 输入: (64,64,1,3) = [rows=Y, cols=X, Z=1, (power, gx, gy)]
    x_phys = np.stack([power, gx, gy], axis=-1)[:, :, np.newaxis, :]
    xt = torch.from_numpy(x_phys).unsqueeze(0)
    with torch.no_grad():
        pred_k = y_norm.inverse(model(x_norm.forward(xt)))[0, :, :, 0].numpy()
    return (pred_k - 273.15).astype(np.float32)


def metrics(grid: np.ndarray) -> dict:
    flat = np.sort(grid.flatten())[::-1]
    return {
        "tmax":  float(flat[0]),
        "tmax50": float(flat[:50].mean()),
        "tmean": float(grid.mean()),
        "tmin":  float(flat[-1]),
    }


# ---------------------------------------------------------------------------
# 布局加载
# ---------------------------------------------------------------------------
def load_milp(case_name: str) -> Layout:
    d = json.load(open(MILP_DIR / case_name / "layout.json"))
    return Layout(placements=[
        Placement(chiplet_id=c["name"],
                  x=c["x"] * SCALE, y=c["y"] * SCALE,
                  rotation=int(round(math.degrees(c["angle_rad"]))) % 360)
        for c in d["chiplets"]
    ])


def load_custom(path: str) -> Layout:
    d = json.load(open(path))
    placements = []
    for c in d.get("chiplets", []):
        # 支持 summary.json 格式（x_mm/y_mm）和 layout.json 格式（x/y in μm）
        if "x_mm" in c:
            x, y = c["x_mm"], c["y_mm"]
        else:
            x, y = c["x"] * SCALE, c["y"] * SCALE
        rot = c.get("rotation", int(round(math.degrees(c.get("angle_rad", 0)))) % 360)
        placements.append(Placement(chiplet_id=c["name"], x=x, y=y, rotation=rot))
    return Layout(placements=placements)


# ---------------------------------------------------------------------------
# 主逻辑
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="UFNO thermal simulation test")
    parser.add_argument("--cases", nargs="+", default=["Case3"],
                        help="Case names to test (default: Case3)")
    parser.add_argument("--hotspot", action="store_true",
                        help="Also run HotSpot and compare")
    parser.add_argument("--layout", default=None,
                        help="Custom layout JSON (overrides MILP, only for single case)")
    args = parser.parse_args()

    if args.hotspot:
        from thermopt.thermal.hotspot import HotSpotBackend

    print(f"\n{'Case':8} {'Tmax':>8} {'T50':>8} {'Tmean':>8} {'Tmin':>8}"
          + ("  |  HS_Tmax  HS_T50  ΔTmax" if args.hotspot else ""))
    print("-" * (50 + (30 if args.hotspot else 0)))

    for case_name in args.cases:
        ci = load_atplace_case(CASES_DIR / case_name, {}, 42)
        case = ci.case

        layout = (load_custom(args.layout)
                  if args.layout and len(args.cases) == 1
                  else load_milp(case_name))

        t0 = time.time()
        grid = ufno_predict(case, layout)
        m = metrics(grid)
        t_ufno = time.time() - t0

        row = (f"{case_name:8} {m['tmax']:>8.2f} {m['tmax50']:>8.2f} "
               f"{m['tmean']:>8.2f} {m['tmin']:>8.2f}  ({t_ufno:.1f}s)")

        if args.hotspot:
            hs = HotSpotBackend(case=case,
                                config={"hotspot_binary": str(HOTSPOT_BIN)})
            t0 = time.time()
            hs_grid = hs.simulate(case, layout)
            mh = metrics(hs_grid)
            t_hs = time.time() - t0
            row += (f"  |  {mh['tmax']:>7.2f}  {mh['tmax50']:>6.2f}"
                    f"  {m['tmax']-mh['tmax']:>+6.2f}  ({t_hs:.0f}s)")

        print(row)

    print()


if __name__ == "__main__":
    main()
