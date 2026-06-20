from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Callable

import numpy as np

from thermopt.layout.geometry import hpwl, total_outline_penalty, total_overlap_penalty
from thermopt.layout.objects import FloorplanCase, Layout, Placement
from thermopt.objective.cost import CostResult
from thermopt.optimizer.sequence_pair import decode_sequence_pair


ROTATIONS = (0, 90, 180, 270)


@dataclass(frozen=True)
class ATPlaceWLResult:
    best_layout: Layout
    best_cost: CostResult
    best_curve: list[float]
    phases: list[dict[str, float | str | bool]]
    solver_success: bool
    solver_message: str
    solver_objective: float


def optimize(
    case: FloorplanCase,
    initial_layout: Layout,
    objective: Callable[[Layout], CostResult],
    config: dict,
    seed: int,
) -> ATPlaceWLResult:
    rng = np.random.default_rng(seed)
    phases: list[dict[str, float | str | bool]] = []

    best_layout = initial_layout
    best_cost = objective(initial_layout)
    best_curve = [best_cost.total]
    _record(phases, "initial", best_layout, best_cost, case)

    milp_layout, solver_success, solver_message, solver_objective = _solve_clump_milp(case, config)
    milp_cost = objective(milp_layout)
    best_layout, best_cost = _choose_better_legal(case, best_layout, best_cost, milp_layout, milp_cost)
    best_curve.append(best_cost.total)
    _record(phases, "clump_milp", milp_layout, milp_cost, case)

    refined_layout = _analytical_refine(case, best_layout, config, seed + 17)
    refined_cost = objective(refined_layout)
    legal_refined = _legalize_by_sequence_pair(case, refined_layout)
    legal_refined_cost = objective(legal_refined)
    best_layout, best_cost = _choose_better_legal(case, best_layout, best_cost, legal_refined, legal_refined_cost)
    best_curve.append(best_cost.total)
    _record(phases, "analytical_refine", legal_refined, legal_refined_cost, case)

    perturbed_layout, perturbed_cost, perturb_curve = _legal_perturb(case, best_layout, objective, config, rng)
    best_layout, best_cost = _choose_better_legal(case, best_layout, best_cost, perturbed_layout, perturbed_cost)
    best_curve.extend(perturb_curve)
    _record(phases, "legal_perturb", perturbed_layout, perturbed_cost, case)

    return ATPlaceWLResult(
        best_layout=best_layout,
        best_cost=best_cost,
        best_curve=best_curve,
        phases=phases,
        solver_success=solver_success,
        solver_message=solver_message,
        solver_objective=solver_objective,
    )


def _solve_clump_milp(case: FloorplanCase, config: dict) -> tuple[Layout, bool, str, float]:
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import lil_matrix
    except ImportError as exc:
        raise RuntimeError("atplace optimizer requires scipy.optimize.milp") from exc

    time_limit = float(config.get("milp_time_limit", 120.0))
    mip_rel_gap = float(config.get("mip_rel_gap", 0.001))
    verbose = bool(config.get("verbose", False))

    ids = [chiplet.id for chiplet in case.chiplets]
    index = {chiplet_id: i for i, chiplet_id in enumerate(ids)}
    n = len(ids)
    pair_terms = _pair_clumps(case, index)
    p = len(pair_terms)
    geom_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    x0 = 0
    y0 = n
    o0 = 2 * n
    tx0 = o0 + 4 * n
    ty0 = tx0 + p
    b0 = ty0 + p
    num_vars = b0 + 4 * len(geom_pairs)

    c = np.zeros(num_vars)
    for k, term in enumerate(pair_terms):
        c[tx0 + k] = term.weight
        c[ty0 + k] = term.weight

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

    def add(coefficients: dict[int, float], lo: float = -np.inf, hi: float = np.inf) -> None:
        rows.append(coefficients)
        lows.append(lo)
        highs.append(hi)

    for i in range(n):
        add({o0 + 4 * i + r: 1.0 for r in range(4)}, lo=1.0, hi=1.0)
        add({x0 + i: -1.0, **{o0 + 4 * i + r: 0.5 * _rotated_size(case, ids[i], ROTATIONS[r])[0] for r in range(4)}}, hi=0.0)
        add({x0 + i: 1.0, **{o0 + 4 * i + r: 0.5 * _rotated_size(case, ids[i], ROTATIONS[r])[0] for r in range(4)}}, hi=case.outline_width)
        add({y0 + i: -1.0, **{o0 + 4 * i + r: 0.5 * _rotated_size(case, ids[i], ROTATIONS[r])[1] for r in range(4)}}, hi=0.0)
        add({y0 + i: 1.0, **{o0 + 4 * i + r: 0.5 * _rotated_size(case, ids[i], ROTATIONS[r])[1] for r in range(4)}}, hi=case.outline_height)

    for k, term in enumerate(pair_terms):
        coeff_x = {x0 + term.left: 1.0, x0 + term.right: -1.0, tx0 + k: -1.0}
        coeff_y = {y0 + term.left: 1.0, y0 + term.right: -1.0, ty0 + k: -1.0}
        for r, rotation in enumerate(ROTATIONS):
            ox, oy = _rotate_offset(term.left_offset[0], term.left_offset[1], rotation)
            coeff_x[o0 + 4 * term.left + r] = coeff_x.get(o0 + 4 * term.left + r, 0.0) + ox
            coeff_y[o0 + 4 * term.left + r] = coeff_y.get(o0 + 4 * term.left + r, 0.0) + oy
            ox, oy = _rotate_offset(term.right_offset[0], term.right_offset[1], rotation)
            coeff_x[o0 + 4 * term.right + r] = coeff_x.get(o0 + 4 * term.right + r, 0.0) - ox
            coeff_y[o0 + 4 * term.right + r] = coeff_y.get(o0 + 4 * term.right + r, 0.0) - oy
        add(coeff_x, hi=0.0)
        add({col: -value for col, value in coeff_x.items() if col != tx0 + k} | {tx0 + k: -1.0}, hi=0.0)
        add(coeff_y, hi=0.0)
        add({col: -value for col, value in coeff_y.items() if col != ty0 + k} | {ty0 + k: -1.0}, hi=0.0)

    big_m = max(case.outline_width, case.outline_height) + 2.0 * max(max(c.width, c.height) for c in case.chiplets)
    for pair_index, (i, j) in enumerate(geom_pairs):
        s_left = b0 + 4 * pair_index
        s_right = s_left + 1
        s_below = s_left + 2
        s_above = s_left + 3
        add(_separation_row(case, ids, o0, x0 + i, x0 + j, i, j, "x", s_left, -big_m), hi=0.0)
        add(_separation_row(case, ids, o0, x0 + j, x0 + i, j, i, "x", s_right, -big_m), hi=0.0)
        add(_separation_row(case, ids, o0, y0 + i, y0 + j, i, j, "y", s_below, -big_m), hi=0.0)
        add(_separation_row(case, ids, o0, y0 + j, y0 + i, j, i, "y", s_above, -big_m), hi=0.0)
        add({s_left: 1.0, s_right: 1.0, s_below: 1.0, s_above: 1.0}, hi=3.0)

    matrix = lil_matrix((len(rows), num_vars))
    for row_index, row in enumerate(rows):
        for col, value in row.items():
            matrix[row_index, col] = value

    result = milp(
        c,
        integrality=integrality,
        bounds=Bounds(lb, ub),
        constraints=LinearConstraint(matrix.tocsr(), np.array(lows), np.array(highs)),
        options={"time_limit": time_limit, "mip_rel_gap": mip_rel_gap, "disp": verbose},
    )
    if result.x is None:
        return _centered_grid_layout(case), False, str(result.message), float("inf")
    return _layout_from_milp(case, ids, result.x, x0, y0, o0), bool(result.success), str(result.message), float(result.fun)


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
        offsets = net.pin_offsets or tuple((0.0, 0.0) for _ in net.chiplets)
        for a in range(len(net.chiplets)):
            for b in range(a + 1, len(net.chiplets)):
                left = index[net.chiplets[a]]
                right = index[net.chiplets[b]]
                left_offset = offsets[a]
                right_offset = offsets[b]
                if left > right:
                    left, right = right, left
                    left_offset, right_offset = right_offset, left_offset
                values = accum.setdefault((left, right), [0.0, 0.0, 0.0, 0.0, 0.0])
                values[0] += 1.0
                values[1] += left_offset[0]
                values[2] += left_offset[1]
                values[3] += right_offset[0]
                values[4] += right_offset[1]
    terms = []
    for (left, right), values in sorted(accum.items()):
        weight = values[0]
        terms.append(_PairTerm(left, right, weight, (values[1] / weight, values[2] / weight), (values[3] / weight, values[4] / weight)))
    return terms


def _separation_row(
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
    row = {first_var: 1.0, second_var: -1.0, switch_var: switch_coeff}
    axis_index = 0 if axis == "x" else 1
    for r, rotation in enumerate(ROTATIONS):
        row[o0 + 4 * first + r] = row.get(o0 + 4 * first + r, 0.0) + 0.5 * _rotated_size(case, ids[first], rotation)[axis_index]
        row[o0 + 4 * second + r] = row.get(o0 + 4 * second + r, 0.0) + 0.5 * _rotated_size(case, ids[second], rotation)[axis_index]
    return row


def _analytical_refine(case: FloorplanCase, layout: Layout, config: dict, seed: int) -> Layout:
    steps = int(config.get("refine_steps", 600))
    if steps <= 0:
        return layout
    try:
        import torch
    except ImportError:
        return layout

    torch.manual_seed(seed)
    ids = [placement.chiplet_id for placement in layout.placements]
    centers = np.array([_placement_center(case, placement) for placement in layout.placements], dtype=np.float64)
    rotations = {placement.chiplet_id: placement.rotation for placement in layout.placements}
    widths = np.array([_rotated_size(case, chiplet_id, rotations[chiplet_id])[0] for chiplet_id in ids], dtype=np.float64)
    heights = np.array([_rotated_size(case, chiplet_id, rotations[chiplet_id])[1] for chiplet_id in ids], dtype=np.float64)

    xy = torch.tensor(centers, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([xy], lr=float(config.get("learning_rate", 0.02)))
    overlap_weight = float(config.get("density_weight", 5000.0))
    outline_weight = float(config.get("outline_weight", 20000.0))
    wl_weight = float(config.get("wl_weight", 1.0))

    widths_t = torch.tensor(widths, dtype=torch.float64)
    heights_t = torch.tensor(heights, dtype=torch.float64)
    net_tensors = _net_tensors(case, ids, rotations)

    best_layout = layout
    best_score = _legal_hpwl_score(case, layout)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = wl_weight * _smooth_hpwl_torch(xy, net_tensors)
        loss = loss + outline_weight * _outline_torch(xy, widths_t, heights_t, case.outline_width, case.outline_height)
        loss = loss + overlap_weight * _overlap_torch(xy, widths_t, heights_t)
        loss.backward()
        optimizer.step()

        candidate = _layout_from_centers(case, ids, xy.detach().numpy(), rotations)
        candidate_score = _legal_hpwl_score(case, candidate)
        if candidate_score < best_score:
            best_layout = candidate
            best_score = candidate_score
    return best_layout


def _legal_perturb(
    case: FloorplanCase,
    layout: Layout,
    objective: Callable[[Layout], CostResult],
    config: dict,
    rng: np.random.Generator,
) -> tuple[Layout, CostResult, list[float]]:
    iterations = int(config.get("legal_perturb_iterations", 2500))
    if iterations <= 0:
        cost = objective(layout)
        return layout, cost, [cost.total]

    ids = [placement.chiplet_id for placement in layout.placements]
    rotations = {placement.chiplet_id: placement.rotation for placement in layout.placements}
    positive, negative = _sequence_pair_from_layout(case, layout)
    current = decode_sequence_pair(case, positive, negative, rotations)
    current_cost = objective(current)
    best = current
    best_cost = current_cost
    curve = [best_cost.total]
    initial_temp = float(config.get("initial_anneal_temp", 0.4))
    final_temp = float(config.get("final_anneal_temp", 0.002))
    report_every = max(1, int(config.get("report_every", 100)))

    for step in range(iterations):
        frac = step / max(1, iterations - 1)
        anneal_temp = initial_temp * ((final_temp / initial_temp) ** frac)
        cand_positive = list(positive)
        cand_negative = list(negative)
        cand_rotations = dict(rotations)
        move = str(rng.choice(["swap_positive", "swap_negative", "swap_both", "rotate"], p=[0.32, 0.32, 0.24, 0.12]))
        if move == "rotate":
            chiplet_id = str(rng.choice(ids))
            cand_rotations[chiplet_id] = int(rng.choice(ROTATIONS))
        else:
            _swap_two(cand_positive if move in {"swap_positive", "swap_both"} else cand_negative, rng)
            if move == "swap_both":
                _swap_two(cand_negative, rng)
        candidate = decode_sequence_pair(case, cand_positive, cand_negative, cand_rotations)
        candidate_cost = objective(candidate)
        delta = candidate_cost.total - current_cost.total
        if delta <= 0.0 or rng.random() < exp(-delta / max(anneal_temp, 1e-12)):
            positive = cand_positive
            negative = cand_negative
            rotations = cand_rotations
            current = candidate
            current_cost = candidate_cost
            if current_cost.total < best_cost.total:
                best = current
                best_cost = current_cost
        if step % report_every == 0 or step == iterations - 1:
            curve.append(best_cost.total)
    return best, best_cost, curve


def _legalize_by_sequence_pair(case: FloorplanCase, layout: Layout) -> Layout:
    rotations = {placement.chiplet_id: placement.rotation for placement in layout.placements}
    centers = {placement.chiplet_id: _placement_center(case, placement) for placement in layout.placements}
    candidates = [layout]

    orderings = [
        (
            sorted(centers, key=lambda chiplet_id: (centers[chiplet_id][0] + centers[chiplet_id][1], centers[chiplet_id][0])),
            sorted(centers, key=lambda chiplet_id: (centers[chiplet_id][0] - centers[chiplet_id][1], centers[chiplet_id][0])),
        ),
        (
            sorted(centers, key=lambda chiplet_id: (centers[chiplet_id][0], centers[chiplet_id][1])),
            sorted(centers, key=lambda chiplet_id: (centers[chiplet_id][1], centers[chiplet_id][0])),
        ),
        (
            sorted(centers, key=lambda chiplet_id: (centers[chiplet_id][1], centers[chiplet_id][0])),
            sorted(centers, key=lambda chiplet_id: (centers[chiplet_id][0], centers[chiplet_id][1])),
        ),
    ]
    for positive, negative in orderings:
        candidates.append(decode_sequence_pair(case, positive, negative, rotations))

    legal = [
        candidate
        for candidate in candidates
        if total_overlap_penalty(case, candidate) <= 1e-9 and total_outline_penalty(case, candidate) <= 1e-9
    ]
    if not legal:
        return min(candidates, key=lambda candidate: _legal_hpwl_score(case, candidate))
    return min(legal, key=lambda candidate: hpwl(case, candidate))


def _sequence_pair_from_layout(case: FloorplanCase, layout: Layout) -> tuple[list[str], list[str]]:
    centers = {placement.chiplet_id: _placement_center(case, placement) for placement in layout.placements}
    positive = sorted(centers, key=lambda chiplet_id: (centers[chiplet_id][0] + centers[chiplet_id][1], centers[chiplet_id][0]))
    negative = sorted(centers, key=lambda chiplet_id: (centers[chiplet_id][0] - centers[chiplet_id][1], centers[chiplet_id][0]))
    return positive, negative


def _choose_better_legal(
    case: FloorplanCase,
    left_layout: Layout,
    left_cost: CostResult,
    right_layout: Layout,
    right_cost: CostResult,
) -> tuple[Layout, CostResult]:
    left_legal = total_overlap_penalty(case, left_layout) <= 1e-9 and total_outline_penalty(case, left_layout) <= 1e-9
    right_legal = total_overlap_penalty(case, right_layout) <= 1e-9 and total_outline_penalty(case, right_layout) <= 1e-9
    if right_legal and (not left_legal or right_cost.metrics["wirelength"] < left_cost.metrics["wirelength"]):
        return right_layout, right_cost
    return left_layout, left_cost


def _record(phases: list[dict[str, float | str | bool]], name: str, layout: Layout, cost: CostResult, case: FloorplanCase) -> None:
    phases.append(
        {
            "phase": name,
            "wirelength": float(cost.metrics["wirelength"]),
            "wirelength_m": float(cost.metrics["wirelength"] * 0.001),
            "total_cost": float(cost.total),
            "overlap_penalty": float(total_overlap_penalty(case, layout)),
            "outline_penalty": float(total_outline_penalty(case, layout)),
        }
    )


def _net_tensors(case: FloorplanCase, ids: list[str], rotations: dict[str, int]):
    import torch

    index = {chiplet_id: i for i, chiplet_id in enumerate(ids)}
    max_degree = max((len(net.chiplets) for net in case.nets), default=2)
    indices = []
    offsets = []
    mask = []
    for net in case.nets:
        net_indices = [index[chiplet_id] for chiplet_id in net.chiplets]
        net_offsets = []
        raw_offsets = net.pin_offsets or tuple((0.0, 0.0) for _ in net.chiplets)
        for chiplet_id, raw_offset in zip(net.chiplets, raw_offsets):
            net_offsets.append(_rotate_offset(raw_offset[0], raw_offset[1], rotations[chiplet_id]))
        while len(net_indices) < max_degree:
            net_indices.append(0)
            net_offsets.append((0.0, 0.0))
        indices.append(net_indices)
        offsets.append(net_offsets)
        mask.append([1.0] * len(net.chiplets) + [0.0] * (max_degree - len(net.chiplets)))
    return (
        torch.tensor(indices, dtype=torch.long),
        torch.tensor(offsets, dtype=torch.float64),
        torch.tensor(mask, dtype=torch.float64),
    )


def _smooth_hpwl_torch(xy, net_tensors, gamma: float = 0.25):
    import torch

    indices, offsets, mask = net_tensors
    pins = xy[indices] + offsets
    invalid_low = (1.0 - mask) * -1e6
    invalid_high = (1.0 - mask) * 1e6
    x = pins[:, :, 0]
    y = pins[:, :, 1]
    xmax = gamma * torch.logsumexp((x + invalid_low) / gamma, dim=1)
    xmin = -gamma * torch.logsumexp((-x - invalid_high) / gamma, dim=1)
    ymax = gamma * torch.logsumexp((y + invalid_low) / gamma, dim=1)
    ymin = -gamma * torch.logsumexp((-y - invalid_high) / gamma, dim=1)
    return (xmax - xmin + ymax - ymin).sum()


def _outline_torch(xy, widths, heights, outline_width: float, outline_height: float):
    import torch

    return (
        torch.relu(widths * 0.5 - xy[:, 0]).square()
        + torch.relu(xy[:, 0] + widths * 0.5 - outline_width).square()
        + torch.relu(heights * 0.5 - xy[:, 1]).square()
        + torch.relu(xy[:, 1] + heights * 0.5 - outline_height).square()
    ).sum()


def _overlap_torch(xy, widths, heights):
    import torch

    overlap = xy.new_tensor(0.0)
    for i in range(len(widths)):
        dx = torch.relu((widths[i] + widths[i + 1 :]) * 0.5 - torch.abs(xy[i, 0] - xy[i + 1 :, 0]))
        dy = torch.relu((heights[i] + heights[i + 1 :]) * 0.5 - torch.abs(xy[i, 1] - xy[i + 1 :, 1]))
        overlap = overlap + (dx * dy).sum()
    return overlap


def _legal_hpwl_score(case: FloorplanCase, layout: Layout) -> float:
    overlap = total_overlap_penalty(case, layout)
    outline = total_outline_penalty(case, layout)
    return hpwl(case, layout) + 1e9 * (overlap + outline)


def _layout_from_milp(case: FloorplanCase, ids: list[str], values: np.ndarray, x0: int, y0: int, o0: int) -> Layout:
    placements = []
    for i, chiplet_id in enumerate(ids):
        orientation_values = values[o0 + 4 * i : o0 + 4 * i + 4]
        rotation = ROTATIONS[int(np.argmax(orientation_values))]
        width, height = _rotated_size(case, chiplet_id, rotation)
        placements.append(Placement(chiplet_id, float(values[x0 + i] - width * 0.5), float(values[y0 + i] - height * 0.5), rotation))
    return Layout(tuple(placements))


def _layout_from_centers(case: FloorplanCase, ids: list[str], centers: np.ndarray, rotations: dict[str, int]) -> Layout:
    placements = []
    for chiplet_id, (cx, cy) in zip(ids, centers):
        width, height = _rotated_size(case, chiplet_id, rotations[chiplet_id])
        placements.append(Placement(chiplet_id, float(cx - width * 0.5), float(cy - height * 0.5), rotations[chiplet_id]))
    return Layout(tuple(placements))


def _centered_grid_layout(case: FloorplanCase) -> Layout:
    placements = []
    x = y = row_height = 0.0
    for chiplet in case.chiplets:
        if x + chiplet.width > case.outline_width:
            x = 0.0
            y += row_height
            row_height = 0.0
        placements.append(Placement(chiplet.id, x, y))
        x += chiplet.width
        row_height = max(row_height, chiplet.height)
    return Layout(tuple(placements))


def _placement_center(case: FloorplanCase, placement: Placement) -> tuple[float, float]:
    width, height = placement.rotated_size(case.chiplet_by_id[placement.chiplet_id])
    return placement.x + width * 0.5, placement.y + height * 0.5


def _rotated_size(case: FloorplanCase, chiplet_id: str, rotation: int) -> tuple[float, float]:
    chiplet = case.chiplet_by_id[chiplet_id]
    if rotation % 180 == 90:
        return chiplet.height, chiplet.width
    return chiplet.width, chiplet.height


def _rotate_offset(offset_x: float, offset_y: float, rotation: int) -> tuple[float, float]:
    rotation = rotation % 360
    if rotation == 0:
        return offset_x, offset_y
    if rotation == 90:
        return -offset_y, offset_x
    if rotation == 180:
        return -offset_x, -offset_y
    if rotation == 270:
        return offset_y, -offset_x
    raise ValueError(f"unsupported rotation: {rotation}")


def _swap_two(values: list[str], rng: np.random.Generator) -> None:
    if len(values) < 2:
        return
    i, j = rng.choice(len(values), size=2, replace=False)
    values[int(i)], values[int(j)] = values[int(j)], values[int(i)]
