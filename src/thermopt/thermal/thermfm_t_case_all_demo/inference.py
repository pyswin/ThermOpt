"""Therm-FM T (case_all) — minimal self-contained inference demo.

Reads input samples from inputs/ (physical values: chiplet power + grid
coordinates, shape 3x64x64), normalizes them, runs the ScOT model, and writes
the predicted steady-state temperature field in Kelvin (64x64) to outputs/.

Self-contained: the model definition ships in ./model.py next to this script.
External dependencies: torch, transformers, numpy.

Usage:
    python inference.py            # auto-selects GPU if available, else CPU
    python inference.py --gpu      # force GPU
    python inference.py --cpu      # force CPU
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

from model import ScOT  # local, self-contained model definition

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model")
IN_DIR = os.path.join(HERE, "inputs")
OUT_DIR = os.path.join(HERE, "outputs")


def main():
    ap = argparse.ArgumentParser(description="Therm-FM T inference demo.")
    ap.add_argument("--gpu", action="store_true", help="force GPU")
    ap.add_argument("--cpu", action="store_true", help="force CPU")
    args = ap.parse_args()
    if args.cpu:
        device = "cpu"
    elif args.gpu:
        device = "cuda"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- normalization constants (computed on the case_all training split) ----
    with open(os.path.join(MODEL_DIR, "normalization_constants.json")) as f:
        stats = json.load(f)
    mean_in = np.array(stats["input"]["mean"], dtype=np.float32).reshape(-1, 1, 1)   # (3,1,1)
    std_in = np.array(stats["input"]["std"], dtype=np.float32).reshape(-1, 1, 1)
    mean_out = np.array(stats["output"]["mean"], dtype=np.float32).reshape(-1, 1, 1)  # (1,1,1)
    std_out = np.array(stats["output"]["std"], dtype=np.float32).reshape(-1, 1, 1)

    # ---- load model ----
    print(f"[infer] loading model from {MODEL_DIR} (device={device})")
    model = ScOT.from_pretrained(MODEL_DIR).to(device).eval()

    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(IN_DIR, "*.npz")))
    if not files:
        raise FileNotFoundError(f"No .npz inputs found in {IN_DIR}")
    print(f"[infer] {len(files)} input samples")

    for fp in files:
        d = np.load(fp)
        inp_phys = d["input"].astype(np.float32)        # (3,64,64): [power, grid_x, grid_y]
        inp_norm = (inp_phys - mean_in) / std_in        # normalize to training distribution
        x = torch.from_numpy(inp_norm).unsqueeze(0).to(device)  # (1,3,64,64)
        with torch.no_grad():
            pred_norm = model(pixel_values=x).output     # (1,1,64,64) normalized
        # keep the layer dim -> (1,64,64), matches the model's raw output shape
        pred_K = pred_norm[0].cpu().numpy() * std_out + mean_out

        name = os.path.basename(fp)
        np.savez(
            os.path.join(OUT_DIR, name),
            prediction=pred_K,
            pred_min_K=float(pred_K.min()),
            pred_max_K=float(pred_K.max()),
        )
        print(f"  {name}: pred T {pred_K.min():.2f} .. {pred_K.max():.2f} K")
    print(f"[infer] done -> {OUT_DIR}")


if __name__ == "__main__":
    main()
