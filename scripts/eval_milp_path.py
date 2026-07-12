#!/usr/bin/env python3
"""Eval csv-path vs backend-path inference on MILP layouts, against HotSpot GT.

--path csv:     grid_x/grid_y from json_to_augmented_csv csv, power recomputed via
                rasterize_power_channel (same as backend) rather than read from the
                csv's chiplet_power column, which under-counts power at zero-gap
                touching chiplet boundaries (csv and backend paths now agree exactly
                whenever chiplets have a real gap; only the coordinate source differs)
--path backend: input from UFNOThermalBackend._input_phys (rasterize_power_channel,
                transpose-fixed)

Reads layout from --milp_dir/CaseN/summary.json, csv+GT from --csv_dir
(CaseN/sample_000000.csv + CaseN/hotspot_grid.npy). Prints per-case Tmax + RMSE
table and saves a GT|UF|err|SAU|err heatmap (jet / RdBu_r).

Usage:
  python scripts/eval_milp_path.py --path csv
  python scripts/eval_milp_path.py --path backend
"""
import argparse, sys, json, numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for p in [str(ROOT/"src"), str(ROOT/"src/thermopt/thermal/ufno_demo"), str(ROOT/"src/thermopt/thermal/saufno_demo")]:
    if p not in sys.path: sys.path.insert(0, p)
from thermopt.data.atplace import load_atplace_case
from thermopt.layout.objects import Layout, Placement
from thermopt.thermal.ufno import UFNOThermalBackend
from thermopt.thermal.surrogate_input import rasterize_power_channel
import ufno, sau_fno  # noqa: F401
from models.normalize import normalize  # noqa: F401

ap = argparse.ArgumentParser()
ap.add_argument("--path", choices=["csv", "backend"], required=True)
ap.add_argument("--milp_dir", default="atplace/20260704_025209_gurobi_t150s_hotspot")
ap.add_argument("--csv_dir", default="atplace/20260712_180518_from_milp_runs")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--out_dir", default=None)
args = ap.parse_args()
K0 = 273.15
MILP = ROOT/args.milp_dir; CSV = ROOT/args.csv_dir
CASES_DIR = ROOT/"external/ATPlace_pub/cases"
OUT = ROOT/(args.out_dir or f"atplace/thermal_runs/{args.path}_path_{Path(args.milp_dir).name}")
OUT.mkdir(parents=True, exist_ok=True)

models = {}
for tag, demo in [("ufno", ROOT/"src/thermopt/thermal/ufno_demo"), ("saufno", ROOT/"src/thermopt/thermal/saufno_demo")]:
    xn, m, yn = torch.load(f"{demo}/model.pt", map_location="cpu")
    models[tag] = (xn, m.to(args.device).eval(), yn)

def fwd(inp):
    x = torch.tensor(inp).unsqueeze(0).float().to(args.device); out = {}
    for tag, (xn, m, yn) in models.items():
        with torch.no_grad(): o = yn.inverse(m(xn.forward(x)).cpu())
        out[tag] = (o[0,...,0].numpy() - K0).astype(np.float64)
    return out

def csv_input(p, case_obj, layout):
    # grid_x/grid_y come from the csv (training-style coordinates); the power
    # channel is recomputed via rasterize_power_channel (same as the backend
    # path) instead of reading the csv's chiplet_power column directly --
    # that column was written by _chiplet_at_point's first-match-only logic,
    # which under-counts power at zero-gap touching chiplet boundaries
    # (confirmed empirically: backend's summing gives lower RMSE vs HotSpot
    # GT, and the two paths agree exactly whenever chiplets have a real gap).
    df = pd.read_csv(p); fx = np.sort(np.unique(df["grid_x"])); fy = np.sort(np.unique(df["grid_y"]))
    xi = np.searchsorted(fx, df["grid_x"]); yi = np.searchsorted(fy, df["grid_y"])
    grid_x = np.zeros((64,64), dtype=np.float32); grid_y = np.zeros((64,64), dtype=np.float32)
    grid_x[xi, yi] = df["grid_x"]; grid_y[xi, yi] = df["grid_y"]
    power = rasterize_power_channel(case_obj, layout, grid_x, grid_y)
    inp = np.zeros((3,1,64,64), dtype=np.float32)
    inp[0,0] = power; inp[1,0] = grid_x; inp[2,0] = grid_y
    return inp.transpose(3,2,1,0)   # csv path transpose (training axis order)

def align(p, gt):
    rd = np.sqrt(((p-gt)**2).mean()); rt = np.sqrt(((p.T-gt)**2).mean())
    return (p, rd) if rd <= rt else (p.T, rt)

def layout_for(case_obj, cn):
    s = json.load(open(MILP/cn/"summary.json")); pl = []
    for c in s["chiplets"]:
        ch = case_obj.chiplet_by_id[c["name"]]
        w, h = (ch.height, ch.width) if c["rotation"]%180==90 else (ch.width, ch.height)
        pl.append(Placement(c["name"], c["x_mm"]+w*0.5, c["y_mm"]+h*0.5, c["rotation"]))
    return Layout(pl)

cases = [f"Case{i}" for i in range(1,9)]
data = {}
for cn in cases:
    case_obj = load_atplace_case(CASES_DIR/cn, {}, 42).case
    if args.path == "csv":
        pred = fwd(csv_input(CSV/cn/"sample_000000.csv", case_obj, layout_for(case_obj, cn)))
    else:
        be = UFNOThermalBackend(case=case_obj, config={"backend":"ufno","grid_size":[64,64]})
        be._update_power_channel(layout_for(case_obj, cn))
        pred = fwd(be._input_phys)
    gt = np.load(CSV/cn/"hotspot_grid.npy").astype(np.float64)
    data[cn] = {"gt": gt}
    for tag in ["ufno", "saufno"]:
        ap_, rm = align(pred[tag], gt); data[cn][tag] = ap_

print(f"=== {args.path} path | Tmax(GT|UF|SAU) + RMSE ===")
print("{:<6}{:>8}{:>8}{:>9}{:>8}{:>9}{:>9}".format("case","GT","UF","d_UF","SAU","d_SAU","UFrmse"))
rm_u = []
for cn in cases:
    d = data[cn]; gt_t = d["gt"].max(); uf_t = d["ufno"].max(); sau_t = d["saufno"].max()
    ru = np.sqrt(((d["ufno"]-d["gt"])**2).mean()); rm_u.append(ru)
    print("{:<6}{:>8.2f}{:>8.2f}{:>+9.2f}{:>8.2f}{:>+9.2f}{:>9.3f}".format(cn, gt_t, uf_t, uf_t-gt_t, sau_t, sau_t-gt_t, ru))
print("{:<6}{:>8}{:>8}{:>9}{:>8}{:>9}{:>9.3f}".format("mean","","","","","",float(np.mean(rm_u))))

fig, axes = plt.subplots(len(cases), 5, figsize=(18, 3.1*len(cases)))
for j, t in enumerate(["HotSpot GT","U-FNO","Error(GT-UF)","SAU-FNO","Error(GT-SAU)"]):
    axes[0,j].set_title(t, fontsize=10)
for i, cn in enumerate(cases):
    d = data[cn]; gt = d["gt"]; uf = d["ufno"]; sau = d["saufno"]; euf = gt-uf; esau = gt-sau
    vmin = min(gt.min(), uf.min(), sau.min()); vmax = max(gt.max(), uf.max(), sau.max())
    eabs = max(abs(euf).max(), abs(esau).max(), 1.0)
    maps = [gt, uf, euf, sau, esau]; cmaps = ["jet","jet","RdBu_r","jet","RdBu_r"]
    vmins = [vmin,vmin,-eabs,vmin,-eabs]; vmaxs = [vmax,vmax,eabs,vmax,eabs]
    for j in range(5):
        axes[i,j].imshow(maps[j], cmap=cmaps[j], vmin=vmins[j], vmax=vmaxs[j], origin="lower")
        axes[i,j].set_xticks([]); axes[i,j].set_yticks([])
    axes[i,0].set_ylabel(cn, fontsize=9)
fig.suptitle(f"{args.path} path", fontsize=13, y=0.995)
plt.tight_layout(); fig.savefig(OUT/f"{args.path}_path_vs_gt.png", dpi=80)
print(f"fig -> {OUT}/{args.path}_path_vs_gt.png")
