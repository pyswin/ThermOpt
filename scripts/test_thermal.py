"""
独立热仿真测试脚本 — 与梯度/优化无关。

对指定 case 的 MILP 初始布局（或自定义布局）做热预测，支持多后端对照：
  ufno / ufno_demo / thermfm / thermfm_t （统一走 thermopt.thermal.backend.build_thermal_backend）
可选 --hotspot 追加 HotSpot 真值仿真列；可选 --plot 保存温度图 / 误差图 / 多后端对比图。

单位约定（统一）:
  - 温度 °C：UFNO / Therm-FM 后端内部已 Kelvin→°C；HotSpot 解析时已减 273.15。
  - 布局坐标 mm：layout.json 存 μm，读取时 ×0.001 (SCALE)。
  - 输入网格：UFNO 为 (64,64,1,3)；Therm-FM 为 (3,64,64)，均由后端类内部装配。

用法:
    # 默认 = UFNO（等价原脚本行为）
    python3 scripts/test_thermal.py --cases Case3 Case5
    # 单后端
    python3 scripts/test_thermal.py --cases Case3 --backend thermfm
    python3 scripts/test_thermal.py --cases Case3 --backend thermfm --hotspot
    # 多后端 + HotSpot + 出图
    python3 scripts/test_thermal.py --cases Case3 --backends ufno thermfm --hotspot --plot
    python3 scripts/test_thermal.py --cases Case3 Case5 Case6 Case7 --backends ufno thermfm --hotspot --plot
"""
import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/thermopt/thermal/ufno_demo"))

from thermopt.data.atplace import load_atplace_case
from thermopt.layout.objects import Layout, Placement
from thermopt.thermal.backend import build_thermal_backend
from thermopt.thermal.surrogate_input import resample_grid

CASES_DIR   = ROOT / "external/ATPlace_pub/cases"
MILP_DIR    = ROOT / "atplace/20260627_163106_milp150s"
HOTSPOT_BIN = ROOT / "external/ATPlace_pub/thermal/hotspot"
OUTPUT_DIR  = ROOT / "atplace/thermal_test_outputs"
SCALE = 0.001  # layout.json 存 μm → mm

AI_BACKENDS = {"ufno", "ufno_demo", "thermfm", "thermfm_t"}


def plot_compare_multi(cases_data, out_path):
    """visualize.py style, multiple cases in ONE figure. rows = case x backend;
    cols = Ground Truth (HotSpot) | Prediction | Error. Per-case color scale."""
    plt = _mpl()
    row_specs = []
    for (cn, W, H, hs, bes) in cases_data:
        ai = bes[0][1].shape
        hs_r = resample_grid(hs, ai) if hs.shape != ai else hs
        vmin_c = min(float(hs_r.min()), min(float(g.min()) for _, g in bes))
        vmax_c = max(float(hs_r.max()), max(float(g.max()) for _, g in bes))
        for (bn, g) in bes:
            row_specs.append((cn, bn, W, H, hs_r, g, vmin_c, vmax_c))
    n = len(row_specs)
    fig, axes = plt.subplots(n, 3, figsize=(11.5, 3.2 * n), squeeze=False, constrained_layout=True)
    col_titles = ["Ground Truth (HotSpot)", "Prediction", "Error (Pred - GT)"]
    for r, (cn, bn, W, H, hs_r, g, vmin, vmax) in enumerate(row_specs):
        resid = g - hs_r
        emax = max(float(max(abs(resid.min()), abs(resid.max()))), 1e-6)
        im0 = axes[r, 0].imshow(hs_r, origin="lower", extent=[0, W, 0, H], cmap="jet", vmin=vmin, vmax=vmax, aspect="equal")
        im1 = axes[r, 1].imshow(g, origin="lower", extent=[0, W, 0, H], cmap="jet", vmin=vmin, vmax=vmax, aspect="equal")
        im2 = axes[r, 2].imshow(resid, origin="lower", extent=[0, W, 0, H], cmap="RdBu_r", vmin=-emax, vmax=emax, aspect="equal")
        for im, ax, lab in zip((im0, im1, im2), axes[r], ("°C", "°C", "ΔT (°C)")):
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(lab)
        axes[r, 0].set_ylabel(f"{cn} / {bn}", fontsize=9)
        if r == 0:
            for c, t in enumerate(col_titles):
                axes[0, c].set_title(t, fontsize=11)
        axes[r, 2].text(0.02, 0.02, f"max |Δ| {float(np.abs(resid).max()):.2f} °C",
                        transform=axes[r, 2].transAxes, fontsize=8, va="bottom",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
    cases_str = " / ".join(cd[0] for cd in cases_data)
    fig.suptitle(f"{cases_str} — HotSpot vs surrogates (°C)", fontsize=13)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------
def metrics(grid: np.ndarray) -> dict:
    flat = np.sort(grid.flatten())[::-1]
    return {
        "tmax":   float(flat[0]),
        "tmax50": float(flat[:50].mean()),
        "tmean":  float(grid.mean()),
        "tmin":   float(flat[-1]),
    }


# ---------------------------------------------------------------------------
# 布局加载
# ---------------------------------------------------------------------------
def load_milp(case_name: str) -> Layout:
    d = json.load(open(MILP_DIR / case_name / "layout.json"))
    return Layout(placements=[
        Placement(chiplet_id=c["name"],
                  x=c.get("cx_mm", c["x"] * SCALE), y=c.get("cy_mm", c["y"] * SCALE),
                  rotation=int(round(math.degrees(c["angle_rad"]))) % 360)
        for c in d["chiplets"]
    ])


def load_custom(path: str) -> Layout:
    d = json.load(open(path))
    placements = []
    for c in d.get("chiplets", []):
        if "x_mm" in c:                       # summary.json 格式
            x, y = c.get("cx_mm", c["x_mm"]), c.get("cy_mm", c["y_mm"])
        else:                                  # layout.json 格式 (μm)
            x, y = c.get("cx_mm", c["x"] * SCALE), c.get("cy_mm", c["y"] * SCALE)
        rot = c.get("rotation", int(round(math.degrees(c.get("angle_rad", 0)))) % 360)
        placements.append(Placement(chiplet_id=c["name"], x=x, y=y, rotation=rot))
    return Layout(placements=placements)


# ---------------------------------------------------------------------------
# 后端运行（统一接口 build_thermal_backend）
# ---------------------------------------------------------------------------
def _ai_config(backend: str, device: str, thermfm_model: str | None = None) -> dict:
    cfg = {"backend": backend}
    if device and device != "auto":
        cfg["device"] = device
    # 指定外部 Therm-FM 权重目录（含 config.json/normalization_constants.json/pytorch_model.bin），
    # 无需拷贝到 ThermOpt；model.py(ScOT 类) 仍用内置 thermfm_t/，与所有 case1_8_* 架构兼容。
    if backend.startswith("thermfm") and thermfm_model:
        cfg["thermfm_model_dir"] = str(Path(thermfm_model).expanduser().resolve())
        cfg["thermfm_demo_root"] = str((ROOT / "src/thermopt/thermal/thermfm_t").resolve())
    return cfg


def run_ai_backend(backend: str, case, layout, device: str = "auto", thermfm_model: str | None = None):
    """构建并运行 AI 后端，返回 (温度图 °C, 运行秒数)。"""
    be = build_thermal_backend(case, _ai_config(backend, device, thermfm_model))
    t0 = time.time()
    grid = be.simulate(case, layout)
    return np.asarray(grid, dtype=np.float32), time.time() - t0


def run_hotspot(case, layout):
    """构建并运行 HotSpot 后端，返回 (温度图 °C, 运行秒数)。"""
    cfg = {"backend": "hotspot", "hotspot_binary": str(HOTSPOT_BIN)}
    be = build_thermal_backend(case, cfg)
    t0 = time.time()
    grid = be.simulate(case, layout)
    return np.asarray(grid, dtype=np.float32), time.time() - t0


# ---------------------------------------------------------------------------
# 表格打印
# ---------------------------------------------------------------------------
def _fmt(v, w, dec=2):
    return f"{v:>{w}.{dec}f}" if v is not None else f"{'-':>{w}}"


def print_table(rows, with_hotspot: bool) -> None:
    BW, CW, NW = 10, 7, 8
    if with_hotspot:
        hdr = (f"{'Backend':<{BW}} {'Case':<{CW}} "
               f"{'Tmax':>{NW}} {'Tmax50':>{NW}} {'Tmean':>{NW}} {'Tmin':>{NW}} {'Time':>6} | "
               f"{'HS_Tmax':>{NW}} {'HS_Tmax50':>{NW}} {'ΔTmax':>{NW}} {'ΔTmax50':>{NW}} {'HS_Time':>7}")
    else:
        hdr = (f"{'Backend':<{BW}} {'Case':<{CW}} "
               f"{'Tmax':>{NW}} {'Tmax50':>{NW}} {'Tmean':>{NW}} {'Tmin':>{NW}} {'Time':>6}")
    print()
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        line = (f"{r['backend']:<{BW}} {r['case']:<{CW}} "
                f"{_fmt(r['tmax'], NW)} {_fmt(r['tmax50'], NW)} "
                f"{_fmt(r['tmean'], NW)} {_fmt(r['tmin'], NW)} {r['runtime']:>5.1f}s")
        if with_hotspot:
            line += (f" | {_fmt(r['hs_tmax'], NW)} {_fmt(r['hs_tmax50'], NW)} "
                     f"{_fmt(r['dtmax'], NW)} {_fmt(r['dtmax50'], NW)} {r['hs_runtime']:>6.0f}s")
        print(line)
    print()


# ---------------------------------------------------------------------------
# 可视化（仅 --plot 时启用）
# ---------------------------------------------------------------------------
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_single(grid, title, out_path, W, H, vmin=None, vmax=None):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(grid, origin="lower", extent=[0, W, 0, H],
                   cmap="inferno", vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Temperature (°C)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_error(err, title, out_path, W, H, vmax=None):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    amax = (float(np.nanmax(np.abs(err))) if vmax is None else vmax)
    amax = max(amax, 1e-6)
    im = ax.imshow(err, origin="lower", extent=[0, W, 0, H],
                   cmap="RdBu_r", vmin=-amax, vmax=amax, aspect="equal")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("ΔT (°C)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_compare(case_name, hs_grid, be_results, W, H, out_path):
    """visualize.py style: one row per backend; cols = Ground Truth (HotSpot) | Prediction | Error.
    GT/Pred share scale (jet); Error = signed residual (RdBu_r); per-panel colorbar + max|Δ|; no ticks."""
    plt = _mpl()
    n = len(be_results)
    ai_shape = be_results[0][1].shape
    hs_r = resample_grid(hs_grid, ai_shape) if hs_grid.shape != ai_shape else hs_grid
    vmin = min(float(hs_r.min()), min(float(g.min()) for _, g in be_results))
    vmax = max(float(hs_r.max()), max(float(g.max()) for _, g in be_results))
    fig, axes = plt.subplots(n, 3, figsize=(11.5, 3.2 * n), squeeze=False, constrained_layout=True)
    col_titles = ["Ground Truth (HotSpot)", "Prediction", "Error (Pred - GT)"]
    for r, (name, g) in enumerate(be_results):
        resid = g - hs_r
        emax = max(float(max(abs(resid.min()), abs(resid.max()))), 1e-6)
        im0 = axes[r, 0].imshow(hs_r, origin="lower", extent=[0, W, 0, H], cmap="jet", vmin=vmin, vmax=vmax, aspect="equal")
        im1 = axes[r, 1].imshow(g, origin="lower", extent=[0, W, 0, H], cmap="jet", vmin=vmin, vmax=vmax, aspect="equal")
        im2 = axes[r, 2].imshow(resid, origin="lower", extent=[0, W, 0, H], cmap="RdBu_r", vmin=-emax, vmax=emax, aspect="equal")
        for im, ax, lab in zip((im0, im1, im2), axes[r], ("°C", "°C", "ΔT (°C)")):
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(lab)
        axes[r, 0].set_ylabel(name, fontsize=10)
        if r == 0:
            for c, t in enumerate(col_titles):
                axes[0, c].set_title(t, fontsize=11)
        axes[r, 2].text(0.02, 0.02, f"max |Δ| {float(np.abs(resid).max()):.2f} °C",
                        transform=axes[r, 2].transAxes, fontsize=8, va="bottom",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
    fig.suptitle(f"{case_name} — HotSpot vs surrogates (°C)", fontsize=13)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------------------
# 主逻辑
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Thermal simulation test (UFNO / Therm-FM / HotSpot)")
    parser.add_argument("--cases", nargs="+", default=["Case3"],
                        help="Case names (default: Case3)")
    parser.add_argument("--backend", choices=sorted(AI_BACKENDS), default=None,
                        help="Single AI backend: ufno / thermfm / thermfm_t")
    parser.add_argument("--backends", nargs="+", choices=sorted(AI_BACKENDS), default=None,
                        help="Multiple AI backends, e.g. --backends ufno thermfm")
    parser.add_argument("--hotspot", action="store_true",
                        help="Append HotSpot ground-truth comparison columns")
    parser.add_argument("--plot", action="store_true",
                        help="Save temperature / error / comparison maps")
    parser.add_argument("--layout", default=None,
                        help="Custom layout JSON (single case only)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                        help="AI backend device (default: auto → cuda if available)")
    parser.add_argument("--thermfm-model", default=None,
                        help="External Therm-FM checkpoint dir (config.json + normalization_constants.json "
                             "+ pytorch_model.bin). Overrides bundled weights; no copy needed.")
    args = parser.parse_args()

    # 解析后端列表：--backends 优先，其次 --backend，默认 ufno（等价原脚本）
    if args.backends:
        backends = args.backends
    elif args.backend:
        backends = [args.backend]
    else:
        backends = ["ufno"]

    plot_dir = None
    if args.plot:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_dir = OUTPUT_DIR / stamp
        plot_dir.mkdir(parents=True, exist_ok=True)
        print(f"[plot] saving figures to: {plot_dir}")

    if args.thermfm_model:
        print(f"[weights] thermfm: {args.thermfm_model}")

    rows = []
    all_cases_data = []
    for case_name in args.cases:
        ci = load_atplace_case(CASES_DIR / case_name, {}, 42)
        case = ci.case
        W, H = float(case.outline_width), float(case.outline_height)

        layout = (load_custom(args.layout)
                  if args.layout and len(args.cases) == 1
                  else load_milp(case_name))

        # HotSpot 按 case 跑一次，供所有后端对照
        hs_grid, hs_rt, hs_m = None, None, None
        if args.hotspot:
            print(f"[{case_name}] running HotSpot ...", flush=True)
            hs_grid, hs_rt = run_hotspot(case, layout)
            hs_m = metrics(hs_grid)
            print(f"[{case_name}] HotSpot: Tmax={hs_m['tmax']:.2f}°C Tmax50={hs_m['tmax50']:.2f}°C ({hs_rt:.0f}s)")

        be_grids = []
        for be_name in backends:
            grid, rt = run_ai_backend(be_name, case, layout, args.device, args.thermfm_model)
            m = metrics(grid)
            row = {
                "backend": be_name, "case": case_name,
                "tmax": m["tmax"], "tmax50": m["tmax50"],
                "tmean": m["tmean"], "tmin": m["tmin"],
                "runtime": rt,
            }
            if args.hotspot and hs_m is not None:
                row.update(hs_tmax=hs_m["tmax"], hs_tmax50=hs_m["tmax50"],
                           hs_runtime=hs_rt,
                           dtmax=row["tmax"] - hs_m["tmax"],
                           dtmax50=row["tmax50"] - hs_m["tmax50"])
            else:
                row.update(hs_tmax=None, hs_tmax50=None, hs_runtime=None,
                           dtmax=None, dtmax50=None)
            rows.append(row)
            be_grids.append((be_name, grid))
            print(f"[{case_name}] {be_name}: Tmax={m['tmax']:.2f}°C Tmax50={m['tmax50']:.2f}°C ({rt:.1f}s)")

        if args.plot and plot_dir is not None and hs_grid is not None and be_grids:
            all_cases_data.append((case_name, W, H, hs_grid, list(be_grids)))

    if args.plot and plot_dir is not None and all_cases_data:
        if len(all_cases_data) == 1:
            out_png = plot_dir / f"{all_cases_data[0][0]}_compare.png"
        else:
            out_png = plot_dir / "compare.png"
        plot_compare_multi(all_cases_data, out_png)
        print(f"[plot] combined compare -> {out_png}")

    print_table(rows, with_hotspot=args.hotspot)

    if plot_dir is not None:
        with open(plot_dir / "metrics.json", "w") as f:
            json.dump(rows, f, indent=2)
        print(f"[plot] metrics written to: {plot_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
