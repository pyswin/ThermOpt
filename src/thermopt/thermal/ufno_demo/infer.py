"""U-FNO standalone inference demo.

Loads the bundled U-FNO model + normalizers, runs the sample in data/input_sample.npy,
writes the predicted temperature field to data/predicted_output.npy, and prints a
quick comparison against data/output_sample.npy (the ground truth) so the prediction
quality is visible.

Self-contained: depends only on the files in this folder (model.pt, ufno.py,
models/normalize.py) + torch/numpy. No access to the training project or dataset needed.

Run:
    python infer.py
"""
import os
import sys
import numpy as np
import torch

# Make THIS directory importable so the saved model bundle can be unpickled:
#   model class      -> ufno.Net3d                (./ufno.py)
#   normalizer class -> models.normalize.normalize (./models/normalize.py)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ufno import Net3d                  # noqa: F401  (registers module path for torch.load)
from models.normalize import normalize  # noqa: F401

MODEL_PT = os.path.join(HERE, "model.pt")
INPUT_NPY = os.path.join(HERE, "data", "input_sample.npy")
GT_NPY = os.path.join(HERE, "data", "output_sample.npy")
OUT_NPY = os.path.join(HERE, "data", "predicted_output.npy")


def main():
    # 1) Load the model bundle: [x_normalizer, model, y_normalizer]
    x_normalizer, model, y_normalizer = torch.load(MODEL_PT, map_location="cpu")
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[loaded] U-FNO, {n_params/1e6:.2f}M params")

    # 2) Load input: shape (64, 64, 1, 3) = [X, Y, Z=1, C]
    #    C = (chiplet_power [W], grid_x, grid_y)
    x = np.load(INPUT_NPY).astype(np.float32)
    print(f"[input]  shape={x.shape}  "
          f"power[min,max]={x[...,0].min():.1f},{x[...,0].max():.1f}  "
          f"grid_x={x[...,1].min():.1f}..{x[...,1].max():.1f}  "
          f"grid_y={x[...,2].min():.1f}..{x[...,2].max():.1f}")
    x = torch.from_numpy(x).unsqueeze(0)          # add batch dim -> (1,64,64,1,3)

    # 3) Forward: normalize input -> model -> inverse-output to physical units (Kelvin)
    with torch.no_grad():
        xn = x_normalizer.forward(x)              # normalized input
        out = model(xn)                           # (1,64,64,1) normalized prediction
        pred = y_normalizer.inverse(out)          # (1,64,64,1) temperature [Kelvin]
    print(f"[forward] raw model out shape={tuple(out.shape)}  pred shape={tuple(pred.shape)}")

    pred = pred[0, ..., 0].numpy().astype(np.float32)   # (64,64) drop batch & Z
    np.save(OUT_NPY, pred)
    print(f"[output] shape={pred.shape}  predicted K[min,max]={pred.min():.2f},{pred.max():.2f}  -> {OUT_NPY}")

    # 4) Compare with ground truth (sanity check)
    if os.path.exists(GT_NPY):
        gt = np.load(GT_NPY).astype(np.float32).squeeze()   # (64,64)
        err = pred - gt
        rmse = float(np.sqrt(np.mean(err**2)))
        mae = float(np.mean(np.abs(err)))
        maxe = float(np.max(np.abs(err)))
        print(f"[vs GT]  RMSE={rmse:.3f} K   MAE={mae:.3f} K   max|err|={maxe:.3f} K")


if __name__ == "__main__":
    main()
