"""Differentiable thermal loss for gradient-based layout optimization.

Supports multiple surrogate models:
  - "ufno"  : U-FNO (ufno_demo/model.pt), input (1,H,W,1,3), current default
  - "scot"  : ScOT/ThermFM (thermfm_t_case_all_demo/), input (1,3,H,W)  [reserved]
  - "fno"   : plain FNO variant (same .pt format as ufno)                [reserved]

Model is auto-detected from path extension:
  *.pt  file  → ufno/fno branch  (torch.load → x_norm, model, y_norm)
  directory   → scot branch      (HuggingFace from_pretrained)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_MODEL_CACHE: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_thermal_model_for_grad(model_path: str | Path) -> tuple:
    """Load a thermal surrogate model for differentiable inference.

    Returned state tuple is opaque; pass directly to ufno_thermal_loss().
    Cached per resolved model_path.
    """
    import torch
    key = str(Path(model_path).resolve())
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    p = Path(model_path).resolve()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if p.suffix == ".pt":
        # --- UFNO / FNO branch ---
        demo_dir = p.parent
        sys.path.insert(0, str(demo_dir))
        x_norm, model, y_norm = torch.load(str(p), map_location=device, weights_only=False)
        model = model.to(device).eval()
        n = sum(par.numel() for par in model.parameters())
        model_type = "ufno"
        state = (model_type, x_norm, model, y_norm, device)
        print(f"[grad_thermal] {model_type} loaded ({n/1e6:.1f}M params) on {device}: {p.name}")

    else:
        # --- ScOT / ThermFM-L branch (directory with HuggingFace weights) ---
        import json, importlib.util
        model_dir = p
        demo_root = model_dir.parent
        model_py = demo_root / "model.py"
        spec = importlib.util.spec_from_file_location("_thermopt_scot_grad", model_py)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        ScOT = module.ScOT

        with (model_dir / "normalization_constants.json").open() as f:
            stats = json.load(f)

        import torch
        model = ScOT.from_pretrained(str(model_dir)).to(device).eval()
        mean_in  = torch.tensor(stats["input"]["mean"],  dtype=torch.float32, device=device).reshape(-1, 1, 1)
        std_in   = torch.tensor(stats["input"]["std"],   dtype=torch.float32, device=device).reshape(-1, 1, 1)
        mean_out = torch.tensor(stats["output"]["mean"], dtype=torch.float32, device=device).reshape(-1, 1, 1)
        std_out  = torch.tensor(stats["output"]["std"],  dtype=torch.float32, device=device).reshape(-1, 1, 1)
        model_type = "scot"
        state = (model_type, model, mean_in, std_in, mean_out, std_out, device)
        print(f"[grad_thermal] {model_type} loaded on {device}: {model_dir.name}")

    _MODEL_CACHE[key] = state
    return state


# backward-compat alias
def load_scot_for_grad(model_dir: str | Path) -> tuple:
    return load_thermal_model_for_grad(model_dir)


# ---------------------------------------------------------------------------
# Differentiable soft rasterization (shared across model types)
# ---------------------------------------------------------------------------

def soft_rasterize_power_torch(xy, widths_t, heights_t, powers_t, grid_x_t, grid_y_t, sharpness):
    """Differentiable power map construction.

    xy        : [N, 2] float32 tensor (chiplet centers, requires_grad)
    widths_t  : [N] float32
    heights_t : [N] float32
    powers_t  : [N] float32
    grid_x_t  : [H, W] x-coordinates of grid pixels (xy-indexed: varies with cols)
    grid_y_t  : [H, W] y-coordinates (varies with rows)
    sharpness : sigmoid steepness in 1/mm

    Returns: power_map [H, W] float32 with gradient w.r.t. xy.
    """
    import torch
    cx = xy[:, 0].unsqueeze(1).unsqueeze(2)   # [N,1,1]
    cy = xy[:, 1].unsqueeze(1).unsqueeze(2)
    w  = widths_t.unsqueeze(1).unsqueeze(2)
    h  = heights_t.unsqueeze(1).unsqueeze(2)
    p  = powers_t.unsqueeze(1).unsqueeze(2)

    gx = grid_x_t.unsqueeze(0)   # [1,H,W]
    gy = grid_y_t.unsqueeze(0)

    mask_x = (torch.sigmoid(sharpness * (gx - (cx - w * 0.5)))
              * torch.sigmoid(sharpness * ((cx + w * 0.5) - gx)))
    mask_y = (torch.sigmoid(sharpness * (gy - (cy - h * 0.5)))
              * torch.sigmoid(sharpness * ((cy + h * 0.5) - gy)))
    return (p * mask_x * mask_y).sum(dim=0)   # [H, W]


# ---------------------------------------------------------------------------
# Differentiable thermal loss
# ---------------------------------------------------------------------------

def ufno_thermal_loss(
    xy, widths_t, heights_t, powers_t, grid_x_t, grid_y_t,
    model_state: tuple,
    sharpness: float,
    mode: str = "tmax",
    topk: int = 50,
):
    """Differentiable thermal loss.

    Gradient chain: xy → soft_rasterize → model → Tmax/T50.
    Returns scalar tensor on model device; call .double().cpu() before adding
    to the main loss (which lives in float64 on CPU).
    """
    import torch
    model_type = model_state[0]

    if model_type in ("ufno", "fno"):
        _, x_norm, model, y_norm, device = model_state

        xy_f = xy.float().to(device)
        gx   = grid_x_t.to(device)
        gy   = grid_y_t.to(device)
        wf   = widths_t.float().to(device)
        hf   = heights_t.float().to(device)
        pf   = powers_t.float().to(device)

        power_map = soft_rasterize_power_torch(xy_f, wf, hf, pf, gx, gy, sharpness)  # [H,W]

        # UFNO input: (1, H, W, 1, 3) = [batch, rows=Y, cols=X, Z=1, (power,gx,gy)]
        x_in = torch.stack([power_map, gx, gy], dim=-1).unsqueeze(-2).unsqueeze(0)
        xn   = x_norm.forward(x_in)
        out  = model(xn)                              # [1, H, W, 1]
        temp_k = y_norm.inverse(out)[0, :, :, 0]     # [H, W] Kelvin
        temp_c = temp_k - 273.15

    elif model_type == "scot":
        _, model, mean_in, std_in, mean_out, std_out, device = model_state

        xy_f = xy.float().to(device)
        gx   = grid_x_t.to(device)
        gy   = grid_y_t.to(device)
        wf   = widths_t.float().to(device)
        hf   = heights_t.float().to(device)
        pf   = powers_t.float().to(device)

        power_map = soft_rasterize_power_torch(xy_f, wf, hf, pf, gx, gy, sharpness)

        # ScOT input: (1, 3, H, W)
        x_in   = torch.stack([power_map, gx, gy], dim=0).unsqueeze(0)
        x_norm = (x_in - mean_in) / std_in
        pred   = model(pixel_values=x_norm).output   # [1, 1, H, W]
        temp_k = pred * std_out + mean_out
        temp_c = temp_k - 273.15

    else:
        raise ValueError(f"Unknown model_type in model_state: {model_type!r}")

    if mode == "tmax50":
        flat = temp_c.flatten()
        return flat.topk(min(topk, flat.numel())).values.mean()
    return temp_c.max()


# backward-compat alias
def scot_thermal_loss(xy, widths_t, heights_t, powers_t, grid_x_t, grid_y_t,
                      model_state, sharpness, mode="tmax", topk=50):
    return ufno_thermal_loss(xy, widths_t, heights_t, powers_t, grid_x_t, grid_y_t,
                             model_state, sharpness, mode=mode, topk=topk)


def scot_max_temp(xy, widths_t, heights_t, powers_t, grid_x_t, grid_y_t, model_state, sharpness):
    return ufno_thermal_loss(xy, widths_t, heights_t, powers_t, grid_x_t, grid_y_t,
                             model_state, sharpness, mode="tmax")
