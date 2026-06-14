from __future__ import annotations

import numpy as np

from thermopt.layout.geometry import bounds
from thermopt.layout.objects import FloorplanCase, Layout


def rasterize_power(case: FloorplanCase, layout: Layout, grid_size: tuple[int, int]) -> np.ndarray:
    nx, ny = grid_size
    power = np.zeros((ny, nx), dtype=float)
    sx = nx / case.outline_width
    sy = ny / case.outline_height
    chiplets = case.chiplet_by_id
    for placement in layout.placements:
        chiplet = chiplets[placement.chiplet_id]
        x0, y0, x1, y1 = bounds(case, placement)
        ix0 = max(0, min(nx - 1, int(np.floor(x0 * sx))))
        ix1 = max(0, min(nx, int(np.ceil(x1 * sx))))
        iy0 = max(0, min(ny - 1, int(np.floor(y0 * sy))))
        iy1 = max(0, min(ny, int(np.ceil(y1 * sy))))
        if ix1 > ix0 and iy1 > iy0:
            power[iy0:iy1, ix0:ix1] += chiplet.power / ((iy1 - iy0) * (ix1 - ix0))
    return power
