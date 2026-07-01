from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from thermopt.layout.geometry import bounds
from thermopt.layout.objects import FloorplanCase, Layout

SURROGATE_NATIVE_GRID_SIZE = (64, 64)
KELVIN_TO_CELSIUS = 273.15


def layout_signature(case: FloorplanCase, layout: Layout) -> str:
    payload = []
    by_id = layout.by_id
    for chiplet_id in case.chiplet_ids:
        placement = by_id.get(chiplet_id)
        if placement is None:
            raise ValueError(f"layout missing chiplet {chiplet_id}")
        payload.append(
            (
                chiplet_id,
                round(float(placement.x), 6),
                round(float(placement.y), 6),
                int(placement.rotation) % 360,
            )
        )
    return hashlib.sha1(repr(payload).encode("utf-8")).hexdigest()[:16]


def grid_size_from_config(config: dict, default: tuple[int, int] = SURROGATE_NATIVE_GRID_SIZE) -> tuple[int, int]:
    if "grid_size" in config:
        raw = config["grid_size"]
        if isinstance(raw, (list, tuple)) and len(raw) == 2:
            return int(raw[0]), int(raw[1])
    if "num_grid_x" in config and "num_grid_y" in config:
        return int(config["num_grid_x"]), int(config["num_grid_y"])
    return default


def coordinate_grid(case: FloorplanCase, grid_size: tuple[int, int] = SURROGATE_NATIVE_GRID_SIZE) -> tuple[np.ndarray, np.ndarray]:
    nx, ny = int(grid_size[0]), int(grid_size[1])
    xs = np.linspace(0.0, float(case.outline_width), nx, dtype=np.float32)
    ys = np.linspace(0.0, float(case.outline_height), ny, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    return grid_x.astype(np.float32), grid_y.astype(np.float32)


def rasterize_power_channel(
    case: FloorplanCase,
    layout: Layout,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    out: np.ndarray | None = None,
) -> np.ndarray:
    if grid_x.shape != grid_y.shape:
        raise ValueError("grid_x and grid_y must have the same shape")

    power = np.zeros_like(grid_x, dtype=np.float32) if out is None else out
    power.fill(0.0)

    chiplets = case.chiplet_by_id
    eps = 1e-9
    for placement in layout.placements:
        chiplet = chiplets[placement.chiplet_id]
        x0, y0, x1, y1 = bounds(case, placement)
        mask = (
            (grid_x >= x0 - eps)
            & (grid_x <= x1 + eps)
            & (grid_y >= y0 - eps)
            & (grid_y <= y1 + eps)
        )
        if np.any(mask):
            power[mask] += float(chiplet.power)

    return power


def resample_grid(grid: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    target_rows, target_cols = int(target_size[0]), int(target_size[1])
    if grid.shape == (target_rows, target_cols):
        return np.array(grid, copy=True)

    src_rows, src_cols = grid.shape
    src_x = np.linspace(0.0, 1.0, src_cols)
    src_y = np.linspace(0.0, 1.0, src_rows)
    tgt_x = np.linspace(0.0, 1.0, target_cols)
    tgt_y = np.linspace(0.0, 1.0, target_rows)

    row_interp = np.array([np.interp(tgt_x, src_x, row) for row in grid], dtype=float)
    col_interp = np.array([np.interp(tgt_y, src_y, row_interp[:, col]) for col in range(row_interp.shape[1])], dtype=float)
    return col_interp.T

