from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from thermopt.layout.objects import FloorplanCase, Layout, Placement
from thermopt.objective.cost import CostResult


_ROTATIONS = (0, 90, 180, 270)


# ── Phase-1 MILP initialization ───────────────────────────────────────────────

@dataclass(frozen=True)
class MILPInitResult:
    layout: Layout
    solver_success: bool
    solver_message: str
    solver_objective: float


def initialize(case: FloorplanCase, config: dict) -> MILPInitResult:
    """
    Phase-1 MILP initialization.

    Variables
    ---------
    x[i], y[i]  : center coordinates of chiplet i  (continuous)
    o[i, r]     : one-hot rotation indicator, r ∈ {0°,90°,180°,270°}  (binary, sum=1)
    tx[k], ty[k]: linearised |clump_dx|, |clump_dy| per connected pair  (continuous ≥ 0)
    b[p, d]     : non-overlap direction selectors, d ∈ {left,right,below,above}  (binary)

    Objective
    ---------
    Minimise Σ_k weight_k · (tx[k] + ty[k])
    where weight_k = number of nets between pair k, and clump offsets are
    the weighted-average pin positions on each chiplet.

    Constraints
    -----------
    Boundary   : chiplet (with rotated size) stays fully inside the interposer.
    Non-overlap: pairwise big-M; at least one of four directions must hold.
                 RHS = epsilon − min_spacing, so epsilon > 0 allows partial overlap
                 (useful to keep the MILP tractable; analytical refine fixes it later).

    Config keys
    -----------
    milp_time_limit      float  seconds  (default 120)
    mip_rel_gap          float           (default 0.001)
    min_chiplet_spacing  float  mm gap added to the non-overlap requirement  (default 0)
    milp_overlap_epsilon float  mm allowed softening of non-overlap  (default 0)
    verbose              bool            (default False)
    """
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import lil_matrix
    except ImportError as exc:
        raise RuntimeError("milp_wl.initialize requires scipy >= 1.7") from exc

    time_limit = float(config.get("milp_time_limit", 120.0))
    mip_rel_gap = float(config.get("mip_rel_gap", 0.001))
    verbose = bool(config.get("verbose", False))

    problem = build_init_problem(case, config)

    matrix = lil_matrix((len(problem.rows), problem.num_vars))
    for ri, row in enumerate(problem.rows):
        for col, val in row.items():
            matrix[ri, col] = val

    result = milp(
        problem.obj,
        integrality=problem.integrality,
        bounds=Bounds(problem.lb, problem.ub),
        constraints=LinearConstraint(matrix.tocsr(), np.array(problem.lows), np.array(problem.highs)),
        options={"time_limit": time_limit, "mip_rel_gap": mip_rel_gap, "disp": verbose},
    )

    if result.x is None:
        return MILPInitResult(
            layout=_grid_fallback(case),
            solver_success=False,
            solver_message=str(result.message),
            solver_objective=float("inf"),
        )

    layout = _decode_init(case, problem.ids, result.x, problem.x0, problem.y0, problem.o0)
    return MILPInitResult(
        layout=layout,
        solver_success=bool(result.success),
        solver_message=str(result.message),
        solver_objective=float(result.fun),
    )


# ── shared problem construction (consumed by both the scipy/HiGHS backend ────
#    above and the Gurobi snapshot backend in milp_wl_gurobi.py) ─────────────

@dataclass(frozen=True)
class MILPProblem:
    """Backend-agnostic MILP matrix form: min obj·x s.t. lows ≤ rows·x ≤ highs, lb ≤ x ≤ ub."""

    obj: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    integrality: np.ndarray
    rows: list[dict[int, float]]
    lows: list[float]
    highs: list[float]
    num_vars: int
    ids: list[str]
    x0: int
    y0: int
    o0: int


def build_init_problem(case: FloorplanCase, config: dict) -> MILPProblem:
    """Build the Phase-1 MILP matrix form. Shared by every solver backend so that
    a change to the formulation only needs to happen once here."""
    min_spacing = float(config.get("min_chiplet_spacing", 0.0))
    epsilon = float(config.get("milp_overlap_epsilon", 0.0))

    ids = [chiplet.id for chiplet in case.chiplets]
    index = {cid: i for i, cid in enumerate(ids)}
    n = len(ids)
    pair_terms = _pair_clumps(case, index)
    p = len(pair_terms)
    geom_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    pair_count = len(geom_pairs)

    # Variable index offsets
    x0 = 0
    y0 = n
    o0 = 2 * n            # one-hot rotation: 4 vars per chiplet
    tx0 = o0 + 4 * n
    ty0 = tx0 + p
    b0 = ty0 + p
    num_vars = b0 + 4 * pair_count

    # Objective: minimise weighted sum of clump-centre Manhattan distances
    obj = np.zeros(num_vars)
    for k, term in enumerate(pair_terms):
        obj[tx0 + k] = term.weight
        obj[ty0 + k] = term.weight

    # Variable bounds
    lb = np.full(num_vars, -np.inf)
    ub = np.full(num_vars, np.inf)
    lb[x0 : x0 + n] = 0.0
    ub[x0 : x0 + n] = case.outline_width
    lb[y0 : y0 + n] = 0.0
    ub[y0 : y0 + n] = case.outline_height
    lb[o0 : o0 + 4 * n] = 0.0
    ub[o0 : o0 + 4 * n] = 1.0
    lb[tx0 : ty0 + p] = 0.0
    lb[b0:] = 0.0
    ub[b0:] = 1.0

    integrality = np.zeros(num_vars)
    integrality[o0 : o0 + 4 * n] = 1
    integrality[b0:] = 1

    rows: list[dict[int, float]] = []
    lows: list[float] = []
    highs: list[float] = []

    def add(coeffs: dict[int, float], lo: float = -np.inf, hi: float = np.inf) -> None:
        rows.append(coeffs)
        lows.append(lo)
        highs.append(hi)

    # One-hot rotation + boundary constraints
    for i in range(n):
        # Σ o[i,r] = 1
        add({o0 + 4 * i + r: 1.0 for r in range(4)}, lo=1.0, hi=1.0)
        # x[i] - Σ hw_r·o[i,r] ≥ 0   →   -x[i] + Σ hw_r·o[i,r] ≤ 0
        add({x0 + i: -1.0, **{o0 + 4 * i + r: 0.5 * _rotated_wh(case, ids[i], _ROTATIONS[r])[0] for r in range(4)}}, hi=0.0)
        # x[i] + Σ hw_r·o[i,r] ≤ outline_width
        add({x0 + i: 1.0, **{o0 + 4 * i + r: 0.5 * _rotated_wh(case, ids[i], _ROTATIONS[r])[0] for r in range(4)}}, hi=case.outline_width)
        add({y0 + i: -1.0, **{o0 + 4 * i + r: 0.5 * _rotated_wh(case, ids[i], _ROTATIONS[r])[1] for r in range(4)}}, hi=0.0)
        add({y0 + i: 1.0, **{o0 + 4 * i + r: 0.5 * _rotated_wh(case, ids[i], _ROTATIONS[r])[1] for r in range(4)}}, hi=case.outline_height)

    # Clump-centre Manhattan distance   |clump_dx| ≤ tx[k],  |clump_dy| ≤ ty[k]
    for k, term in enumerate(pair_terms):
        cx = {x0 + term.left: 1.0, x0 + term.right: -1.0, tx0 + k: -1.0}
        cy = {y0 + term.left: 1.0, y0 + term.right: -1.0, ty0 + k: -1.0}
        for r, rot in enumerate(_ROTATIONS):
            ox, oy = _rot_offset(term.left_offset[0], term.left_offset[1], rot)
            cx[o0 + 4 * term.left + r] = cx.get(o0 + 4 * term.left + r, 0.0) + ox
            cy[o0 + 4 * term.left + r] = cy.get(o0 + 4 * term.left + r, 0.0) + oy
            ox, oy = _rot_offset(term.right_offset[0], term.right_offset[1], rot)
            cx[o0 + 4 * term.right + r] = cx.get(o0 + 4 * term.right + r, 0.0) - ox
            cy[o0 + 4 * term.right + r] = cy.get(o0 + 4 * term.right + r, 0.0) - oy
        add(cx, hi=0.0)
        add({col: -v for col, v in cx.items() if col != tx0 + k} | {tx0 + k: -1.0}, hi=0.0)
        add(cy, hi=0.0)
        add({col: -v for col, v in cy.items() if col != ty0 + k} | {ty0 + k: -1.0}, hi=0.0)

    # Pairwise non-overlap (big-M), right-hand side = epsilon − min_spacing
    max_size = max(max(chiplet.width, chiplet.height) for chiplet in case.chiplets)
    big_m = max(case.outline_width, case.outline_height) + 2.0 * max_size + min_spacing
    rhs = epsilon - min_spacing

    for pi, (i, j) in enumerate(geom_pairs):
        sl = b0 + 4 * pi      # i left of j
        sr = sl + 1           # j left of i
        sb = sl + 2           # i below j
        sa = sl + 3           # j below i
        add(_sep_row(case, ids, o0, x0 + i, x0 + j, i, j, "x", sl, -big_m), hi=rhs)
        add(_sep_row(case, ids, o0, x0 + j, x0 + i, j, i, "x", sr, -big_m), hi=rhs)
        add(_sep_row(case, ids, o0, y0 + i, y0 + j, i, j, "y", sb, -big_m), hi=rhs)
        add(_sep_row(case, ids, o0, y0 + j, y0 + i, j, i, "y", sa, -big_m), hi=rhs)
        # at least one of the four must hold
        add({sl: 1.0, sr: 1.0, sb: 1.0, sa: 1.0}, hi=3.0)

    return MILPProblem(obj, lb, ub, integrality, rows, lows, highs, num_vars, ids, x0, y0, o0)


def decode_init_solution(case: FloorplanCase, problem: MILPProblem, values: np.ndarray) -> Layout:
    """Decode a raw variable vector (scipy result.x or a Gurobi solution) into a Layout."""
    return _decode_init(case, problem.ids, values, problem.x0, problem.y0, problem.o0)


# ── helpers used only by initialize() ────────────────────────────────────────

@dataclass(frozen=True)
class _PairTerm:
    left: int
    right: int
    weight: float
    left_offset: tuple[float, float]
    right_offset: tuple[float, float]


def _pair_clumps(case: FloorplanCase, index: dict[str, int]) -> list[_PairTerm]:
    accum: dict[tuple[int, int], list[float]] = {}
    for net in case.nets:
        if len(set(net.chiplets)) < 2:
            continue
        offs = net.pin_offsets or tuple((0.0, 0.0) for _ in net.chiplets)
        for a in range(len(net.chiplets)):
            for b in range(a + 1, len(net.chiplets)):
                left, right = index[net.chiplets[a]], index[net.chiplets[b]]
                loff, roff = offs[a], offs[b]
                if left > right:
                    left, right = right, left
                    loff, roff = roff, loff
                v = accum.setdefault((left, right), [0.0, 0.0, 0.0, 0.0, 0.0])
                v[0] += 1.0; v[1] += loff[0]; v[2] += loff[1]; v[3] += roff[0]; v[4] += roff[1]
    return [
        _PairTerm(left, right, v[0], (v[1] / v[0], v[2] / v[0]), (v[3] / v[0], v[4] / v[0]))
        for (left, right), v in sorted(accum.items())
    ]


def _sep_row(
    case: FloorplanCase,
    ids: list[str],
    o0: int,
    first_var: int,
    second_var: int,
    first: int,
    second: int,
    axis: str,
    switch_var: int,
    switch_coeff: float,
) -> dict[int, float]:
    row: dict[int, float] = {first_var: 1.0, second_var: -1.0, switch_var: switch_coeff}
    ai = 0 if axis == "x" else 1
    for r, rot in enumerate(_ROTATIONS):
        row[o0 + 4 * first + r] = row.get(o0 + 4 * first + r, 0.0) + 0.5 * _rotated_wh(case, ids[first], rot)[ai]
        row[o0 + 4 * second + r] = row.get(o0 + 4 * second + r, 0.0) + 0.5 * _rotated_wh(case, ids[second], rot)[ai]
    return row


def _decode_init(case: FloorplanCase, ids: list[str], vals: np.ndarray, x0: int, y0: int, o0: int) -> Layout:
    placements = []
    for i, cid in enumerate(ids):
        rot = _ROTATIONS[int(np.argmax(vals[o0 + 4 * i : o0 + 4 * i + 4]))]
        # vals[x0+i]/vals[y0+i] are already chiplet centers (see MILP variable docs above)
        placements.append(Placement(cid, float(vals[x0 + i]), float(vals[y0 + i]), rot))
    return Layout(tuple(placements))


def _grid_fallback(case: FloorplanCase) -> Layout:
    placements: list[Placement] = []
    x = y = row_h = 0.0
    for chiplet in case.chiplets:
        if x + chiplet.width > case.outline_width:
            x = 0.0; y += row_h; row_h = 0.0
        placements.append(Placement(chiplet.id, x + chiplet.width * 0.5, y + chiplet.height * 0.5))
        x += chiplet.width
        row_h = max(row_h, chiplet.height)
    return Layout(tuple(placements))


def _rotated_wh(case: FloorplanCase, chiplet_id: str, rotation: int) -> tuple[float, float]:
    c = case.chiplet_by_id[chiplet_id]
    return (c.height, c.width) if rotation % 180 == 90 else (c.width, c.height)


def _rot_offset(ox: float, oy: float, rotation: int) -> tuple[float, float]:
    r = rotation % 360
    if r == 0:   return ox, oy
    if r == 90:  return -oy, ox
    if r == 180: return -ox, -oy
    if r == 270: return oy, -ox
    raise ValueError(f"unsupported rotation: {rotation}")


# ── existing WL-MILP optimizer (unchanged) ───────────────────────────────────

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
        placements.append(Placement(chiplet_id, float(values[x0 + i]), float(values[y0 + i]), rotation))

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
