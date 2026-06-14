from __future__ import annotations

import numpy as np

from thermopt.layout.geometry import center
from thermopt.layout.objects import FloorplanCase, Layout


def simulate_temperature(case: FloorplanCase, layout: Layout, config: dict) -> np.ndarray:
    nx, ny = tuple(config.get("grid_size", (100, 80)))
    ambient = float(config.get("ambient", 25.0))
    scale = float(config.get("scale", 1.0))
    sigma_factor = float(config.get("sigma_factor", 1.0))

    xs = np.linspace(0.0, case.outline_width, nx)
    ys = np.linspace(0.0, case.outline_height, ny)
    grid_x, grid_y = np.meshgrid(xs, ys)
    temperature = np.full((ny, nx), ambient, dtype=float)

    chiplets = case.chiplet_by_id
    for placement in layout.placements:
        chiplet = chiplets[placement.chiplet_id]
        cx, cy = center(case, placement)
        width, height = placement.rotated_size(chiplet)
        sigma = max(width, height, 1.0) * sigma_factor
        dist2 = (grid_x - cx) ** 2 + (grid_y - cy) ** 2
        temperature += scale * chiplet.power * np.exp(-dist2 / (2.0 * sigma * sigma))

    return temperature
