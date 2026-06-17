from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from thermopt.layout.objects import FloorplanCase, Layout, Placement
from thermopt.objective.cost import CostResult


@dataclass(frozen=True)
class MILPWLResult:
    best_layout: Layout
    best_cost: CostResult
    best_curve: list[float]
    solver_success: bool
    solver_message: str
    solver_objective: float


def optimize(
    case: FloorplanCase,
    initial_layout: Layout,
    objective: Callable[[Layout], CostResult],
    config: dict,
    seed: int,
) -> MILPWLResult:
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import lil_matrix
    except ImportError as exc:
        raise RuntimeError("milp_wl optimizer requires scipy.optimize.milp") from exc

    allow_rotation = bool(config.get("allow_rotation", True))
    time_limit = float(config.get("time_limit", 120.0))
    mip_rel_gap = float(config.get("mip_rel_gap", 0.01))

    ids = [chiplet.id for chiplet in case.chiplets]
    index = {chiplet_id: i for i, chiplet_id in enumerate(ids)}
    n = len(ids)
    width0 = np.array([chiplet.width for chiplet in case.chiplets], dtype=float)
    height0 = np.array([chiplet.height for chiplet in case.chiplets], dtype=float)
    delta_width = height0 - width0
    delta_height = width0 - height0
    nets = _two_pin_nets(case, index)
    m = len(nets)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    pair_count = len(pairs)

    x0 = 0
    y0 = n
    r0 = 2 * n
    tx0 = 3 * n
    ty0 = 3 * n + m
    b0 = 3 * n + 2 * m
    num_vars = b0 + 4 * pair_count

    c = np.zeros(num_vars)
    c[tx0 : tx0 + m] = 1.0
    c[ty0 : ty0 + m] = 1.0

    lb = np.full(num_vars, -np.inf)
    ub = np.full(num_vars, np.inf)
    for i in range(n):
        lb[x0 + i] = 0.0
        ub[x0 + i] = case.outline_width
        lb[y0 + i] = 0.0
        ub[y0 + i] = case.outline_height
    lb[r0 : r0 + n] = 0.0
    ub[r0 : r0 + n] = 1.0 if allow_rotation else 0.0
    lb[tx0 : ty0 + m] = 0.0
    lb[b0:] = 0.0
    ub[b0:] = 1.0

    integrality = np.zeros(num_vars)
    if allow_rotation:
        integrality[r0 : r0 + n] = 1
    integrality[b0:] = 1

    rows: list[dict[int, float]] = []
    lows: list[float] = []
    highs: list[float] = []

    def add(coefficients: dict[int, float], lo: float = -np.inf, hi: float = np.inf) -> None:
        rows.append(coefficients)
        lows.append(lo)
        highs.append(hi)

    for i in range(n):
        add({x0 + i: -1, r0 + i: delta_width[i] * 0.5}, hi=-width0[i] * 0.5)
        add({x0 + i: 1, r0 + i: delta_width[i] * 0.5}, hi=case.outline_width - width0[i] * 0.5)
        add({y0 + i: -1, r0 + i: delta_height[i] * 0.5}, hi=-height0[i] * 0.5)
        add({y0 + i: 1, r0 + i: delta_height[i] * 0.5}, hi=case.outline_height - height0[i] * 0.5)

    for k, (i, j, ox_i, oy_i, ox_j, oy_j) in enumerate(nets):
        dx_i = -oy_i - ox_i
        dx_j = -oy_j - ox_j
        dy_i = ox_i - oy_i
        dy_j = ox_j - oy_j
        add({x0 + i: 1, x0 + j: -1, r0 + i: dx_i, r0 + j: -dx_j, tx0 + k: -1}, hi=-(ox_i - ox_j))
        add({x0 + i: -1, x0 + j: 1, r0 + i: -dx_i, r0 + j: dx_j, tx0 + k: -1}, hi=ox_i - ox_j)
        add({y0 + i: 1, y0 + j: -1, r0 + i: dy_i, r0 + j: -dy_j, ty0 + k: -1}, hi=-(oy_i - oy_j))
        add({y0 + i: -1, y0 + j: 1, r0 + i: -dy_i, r0 + j: dy_j, ty0 + k: -1}, hi=oy_i - oy_j)

    big_m = max(case.outline_width, case.outline_height) + max(width0.max(), height0.max())
    for pair_index, (i, j) in enumerate(pairs):
        s_left = b0 + 4 * pair_index
        s_right = s_left + 1
        s_below = s_left + 2
        s_above = s_left + 3
        add(
            {x0 + i: 1, x0 + j: -1, r0 + i: delta_width[i] * 0.5, r0 + j: delta_width[j] * 0.5, s_left: -big_m},
            hi=-(width0[i] + width0[j]) * 0.5,
        )
        add(
            {x0 + j: 1, x0 + i: -1, r0 + i: delta_width[i] * 0.5, r0 + j: delta_width[j] * 0.5, s_right: -big_m},
            hi=-(width0[i] + width0[j]) * 0.5,
        )
        add(
            {y0 + i: 1, y0 + j: -1, r0 + i: delta_height[i] * 0.5, r0 + j: delta_height[j] * 0.5, s_below: -big_m},
            hi=-(height0[i] + height0[j]) * 0.5,
        )
        add(
            {y0 + j: 1, y0 + i: -1, r0 + i: delta_height[i] * 0.5, r0 + j: delta_height[j] * 0.5, s_above: -big_m},
            hi=-(height0[i] + height0[j]) * 0.5,
        )
        add({s_left: 1, s_right: 1, s_below: 1, s_above: 1}, hi=3.0)

    matrix = lil_matrix((len(rows), num_vars))
    for row_index, row in enumerate(rows):
        for col, value in row.items():
            matrix[row_index, col] = value

    result = milp(
        c,
        integrality=integrality,
        bounds=Bounds(lb, ub),
        constraints=LinearConstraint(matrix.tocsr(), np.array(lows), np.array(highs)),
        options={"time_limit": time_limit, "mip_rel_gap": mip_rel_gap, "disp": bool(config.get("verbose", False))},
    )

    if result.x is None:
        return MILPWLResult(
            best_layout=initial_layout,
            best_cost=objective(initial_layout),
            best_curve=[objective(initial_layout).total],
            solver_success=False,
            solver_message=str(result.message),
            solver_objective=float("inf"),
        )

    values = result.x
    placements: list[Placement] = []
    for i, chiplet_id in enumerate(ids):
        rotation = int(round(values[r0 + i])) * 90
        width = width0[i] + delta_width[i] * int(round(values[r0 + i]))
        height = height0[i] + delta_height[i] * int(round(values[r0 + i]))
        placements.append(Placement(chiplet_id, float(values[x0 + i] - width * 0.5), float(values[y0 + i] - height * 0.5), rotation))

    layout = Layout(tuple(placements))
    cost = objective(layout)
    return MILPWLResult(
        best_layout=layout,
        best_cost=cost,
        best_curve=[cost.total],
        solver_success=bool(result.success),
        solver_message=str(result.message),
        solver_objective=float(result.fun),
    )


def _two_pin_nets(case: FloorplanCase, index: dict[str, int]) -> list[tuple[int, int, float, float, float, float]]:
    nets = []
    for net in case.nets:
        if len(net.chiplets) != 2 or len(set(net.chiplets)) < 2:
            continue
        offsets = net.pin_offsets or ((0.0, 0.0), (0.0, 0.0))
        nets.append((index[net.chiplets[0]], index[net.chiplets[1]], offsets[0][0], offsets[0][1], offsets[1][0], offsets[1][1]))
    return nets
