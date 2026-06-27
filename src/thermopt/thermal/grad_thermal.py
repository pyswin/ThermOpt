from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_MODEL_CACHE: dict[str, Any] = {}


def load_scot_for_grad(model_dir: str | Path) -> tuple:
    """Load ScOT model weights for differentiable inference. Cached per model_dir."""
    key = str(Path(model_dir).resolve())
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    import torch

    model_dir = Path(model_dir).resolve()
    demo_root = model_dir.parent
    model_py = demo_root / "model.py"

    import importlib.util
    spec = importlib.util.spec_from_file_location("thermopt_thermfm_demo_model_grad", model_py)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    ScOT = module.ScOT

    with (model_dir / "normalization_constants.json").open() as f:
        stats = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ScOT.from_pretrained(str(model_dir)).to(device).eval()

    mean_in  = torch.tensor(stats["input"]["mean"],   dtype=torch.float32, device=device).reshape(-1, 1, 1)
    std_in   = torch.tensor(stats["input"]["std"],    dtype=torch.float32, device=device).reshape(-1, 1, 1)
    mean_out = torch.tensor(stats["output"]["mean"],  dtype=torch.float32, device=device).reshape(-1, 1, 1)
    std_out  = torch.tensor(stats["output"]["std"],   dtype=torch.float32, device=device).reshape(-1, 1, 1)

    state = (model, mean_in, std_in, mean_out, std_out, device)
    _MODEL_CACHE[key] = state
    print(f"[grad_thermal] ScOT loaded on {device}: {model_dir}")
    return state


def soft_rasterize_power_torch(
    xy,
    widths_t,
    heights_t,
    powers_t,
    grid_x_t,
    grid_y_t,
    sharpness: float,
):
    """
    Differentiable power map construction.

    xy        : [N, 2] float32 tensor (chiplet centers, requires_grad)
    widths_t  : [N] float32 tensor (chiplet widths after rotation)
    heights_t : [N] float32 tensor
    powers_t  : [N] float32 tensor
    grid_x_t  : [H, W] float32 tensor (x-coordinates of grid pixels)
    grid_y_t  : [H, W] float32 tensor
    sharpness : sigmoid steepness in 1/length-unit (higher = sharper boundary)

    Returns: power_map [H, W] float32 tensor with gradient w.r.t. xy.
    """
    import torch
    # Broadcast [N,1,1] vs [1,H,W]
    cx = xy[:, 0].unsqueeze(1).unsqueeze(2)
    cy = xy[:, 1].unsqueeze(1).unsqueeze(2)
    w  = widths_t.unsqueeze(1).unsqueeze(2)
    h  = heights_t.unsqueeze(1).unsqueeze(2)
    p  = powers_t.unsqueeze(1).unsqueeze(2)

    gx = grid_x_t.unsqueeze(0)
    gy = grid_y_t.unsqueeze(0)

    mask_x = (
        torch.sigmoid(sharpness * (gx - (cx - w * 0.5)))
        * torch.sigmoid(sharpness * ((cx + w * 0.5) - gx))
    )
    mask_y = (
        torch.sigmoid(sharpness * (gy - (cy - h * 0.5)))
        * torch.sigmoid(sharpness * ((cy + h * 0.5) - gy))
    )
    return (p * mask_x * mask_y).sum(dim=0)  # [H, W]


def scot_thermal_loss(
    xy,
    widths_t,
    heights_t,
    powers_t,
    grid_x_t,
    grid_y_t,
    model_state: tuple,
    sharpness: float,
    mode: str = "tmax",
    topk: int = 50,
) -> Any:
    """
    Differentiable thermal loss via ScOT FNO.

    mode='tmax'  : max temperature (scalar).
    mode='tmax50': mean of top-k hottest pixels (k=topk, default 50).

    xy must be float32 (or will be cast). Returns scalar tensor on model device.
    Gradient flows: xy → soft_rasterize → ScOT → thermal_loss.
    """
    import torch
    model, mean_in, std_in, mean_out, std_out, device = model_state

    xy_f = xy.float().to(device)
    gx   = grid_x_t.to(device)
    gy   = grid_y_t.to(device)
    wf   = widths_t.float().to(device)
    hf   = heights_t.float().to(device)
    pf   = powers_t.float().to(device)

    power_map = soft_rasterize_power_torch(xy_f, wf, hf, pf, gx, gy, sharpness)  # [H, W]

    x_in   = torch.stack([power_map, gx, gy], dim=0).unsqueeze(0)  # [1, 3, H, W]
    x_norm = (x_in - mean_in) / std_in

    pred_norm = model(pixel_values=x_norm).output          # [1, 1, H, W]
    temp_k    = pred_norm * std_out + mean_out             # Kelvin
    temp_c    = temp_k - 273.15                            # Celsius

    if mode == "tmax50":
        flat = temp_c.flatten()
        k = min(topk, flat.numel())
        return flat.topk(k).values.mean()
    return temp_c.max()


def scot_max_temp(xy, widths_t, heights_t, powers_t, grid_x_t, grid_y_t, model_state, sharpness):
    """Backward-compat alias for scot_thermal_loss with mode='tmax'."""
    return scot_thermal_loss(xy, widths_t, heights_t, powers_t, grid_x_t, grid_y_t,
                             model_state, sharpness, mode="tmax")
