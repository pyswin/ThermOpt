"""Eval ThermOpt demo model on a .npy test dataset (default: case1_10_combined_per5_test).

Uses ThermOpt's bundled demo model (saufno_demo / ufno_demo) with the SAME
forward path as infer.py:  x_normalizer.forward -> model -> y_normalizer.inverse.
Reports per-sample RMSE (pred vs GT, Kelvin), grouped by case (outline.npy).

Usage:
  python scripts/eval_per5_test.py --model ufno
  python scripts/eval_per5_test.py --model saufno --device cpu
"""
import sys, argparse, collections
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]  # ThermOpt repo root

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["ufno", "saufno"], default="ufno")
    ap.add_argument("--data", default="/data/sc/menglinghai/data/case1_10_combined_per5_test")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    demo = ROOT / "src/thermopt/thermal" / ("saufno_demo" if args.model == "saufno" else "ufno_demo")
    if str(demo) not in sys.path:
        sys.path.insert(0, str(demo))
    if args.model == "saufno":
        import sau_fno, ufno  # noqa: F401
    else:
        import ufno  # noqa: F401
    from models.normalize import normalize  # noqa: F401

    xn, model, yn = torch.load(demo / "model.pt", map_location="cpu")
    model = model.to(args.device).eval()

    x = torch.tensor(np.load(Path(args.data) / "input.npy")).float()
    y = torch.tensor(np.load(Path(args.data) / "output.npy")).float()
    outline = np.load(Path(args.data) / "outline.npy")
    print(f"[{args.model}] {x.shape[0]} samples  device={args.device}  model={demo/'model.pt'}", flush=True)

    preds = []
    with torch.no_grad():
        z = xn.forward(x).to(args.device)
        for i in range(0, x.shape[0], 20):
            preds.append(yn.inverse(model(z[i:i+20]).cpu()))
    pred = torch.cat(preds, 0)[..., 0].numpy()   # (N,64,64) Kelvin
    gt   = y[..., 0].numpy()
    rmse = np.sqrt(((pred - gt) ** 2).mean(axis=(1, 2)))

    print(f"\n{'case':<10}{'#':<4}{'rmse(K)':<10}")
    groups = collections.OrderedDict()
    for i, s in enumerate(outline):
        groups.setdefault(str(s), []).append(i)
    for s, idxs in sorted(groups.items()):
        for k, i in enumerate(idxs):
            print(f"{s:<10}{k:<4}{rmse[i]:<10.4f}")
        print(f"{s:<10}{'mu':<4}{rmse[idxs].mean():<10.4f}")
    print(f"\noverall rmse = {rmse.mean():.4f} K")

if __name__ == "__main__":
    main()
