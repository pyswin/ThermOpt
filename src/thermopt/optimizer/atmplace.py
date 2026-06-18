from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Callable

import numpy as np

from thermopt.layout.geometry import hpwl, total_outline_penalty, total_overlap_penalty
from thermopt.layout.objects import FloorplanCase, Layout, Placement
from thermopt.objective.cost import CostResult
from thermopt.optimizer import atplace
from thermopt.optimizer.sequence_pair import decode_sequence_pair


ROTATIONS = (0, 90, 180, 270)


@dataclass(frozen=True)
class ATMPlaceResult:
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
) -> ATMPlaceResult:
    rng = np.random.default_rng(seed)
    phases: list[dict[str, float | str | bool]] = []

    best_layout = initial_layout
    best_cost = objective(initial_layout)
    best_curve = [best_cost.total]
    _record(phases, "initial", best_layout, best_cost, case)

    if len(case.chiplets) <= int(config.get("milp_chiplet_limit", 12)):
        seed_config = dict(config)
        seed_config["legal_perturb_iterations"] = 0
        seed_config["refine_steps"] = 0
        seed_result = atplace.optimize(case, initial_layout, objective, seed_config, seed + 11)
        seed_layout = seed_result.best_layout
        solver_success = seed_result.solver_success
        solver_message = seed_result.solver_message
        solver_objective = seed_result.solver_objective
        seed_phase = "milp_seed"
    else:
        seed_layout = _spectral_seed_layout(case)
        solver_success = True
        solver_message = "spectral clump seed"
        solver_objective = hpwl(case, seed_layout)
        seed_phase = "spectral_seed"

    seed_cost = objective(seed_layout)
    best_layout, best_cost = _choose_better_legal(case, best_layout, best_cost, seed_layout, seed_cost)
    best_curve.append(best_cost.total)
    _record(phases, seed_phase, seed_layout, seed_cost, case)

    refined_layout, refine_curve = _cgd_refine(case, best_layout, config, seed + 23)
    refined_layout = _legalize_candidates(case, refined_layout, rng, int(config.get("legalize_candidates", 64)))
    refined_cost = objective(refined_layout)
    best_layout, best_cost = _choose_better_legal(case, best_layout, best_cost, refined_layout, refined_cost)
    best_curve.extend(refine_curve)
    best_curve.append(best_cost.total)
    _record(phases, "orientation_cgd", refined_layout, refined_cost, case)

    polished_layout, polished_cost, polished_curve = _sequence_pair_polish(case, best_layout, objective, config, rng)
    best_layout, best_cost = _choose_better_legal(case, best_layout, best_cost, polished_layout, polished_cost)
    best_curve.extend(polished_curve)
    _record(phases, "legalization", polished_layout, polished_cost, case)

    return ATMPlaceResult(
        best_layout=best_layout,
        best_cost=best_cost,
        best_curve=best_curve,
        phases=phases,
        solver_success=solver_success,
        solver_message=solver_message,
        solver_objective=float(solver_objective),
    )


def _spectral_seed_layout(case: FloorplanCase) -> Layout:
    ids = [chiplet.id for chiplet in case.chiplets]
    index = {chiplet_id: i for i, chiplet_id in enumerate(ids)}
    n = len(ids)
    weights = np.zeros((n, n), dtype=float)
    for net in case.nets:
        unique = list(dict.fromkeys(net.chiplets))
        if len(unique) < 2:
            continue
        weight = 1.0 / max(1, len(unique) - 1)
        for a in range(len(unique)):
            for b in range(a + 1, len(unique)):
                i = index[unique[a]]
                j = index[unique[b]]
                weights[i, j] += weight
                weights[j, i] += weight

    degree = weights.sum(axis=1)
    laplacian = np.diag(degree) - weights
    try:
        _, vectors = np.linalg.eigh(laplacian)
        coords = vectors[:, 1:3] if n >= 3 else np.column_stack([np.arange(n), np.zeros(n)])
    except np.linalg.LinAlgError:
        coords = np.column_stack([np.arange(n), np.zeros(n)])
    if coords.shape[1] < 2:
        coords = np.column_stack([coords[:, 0], np.zeros(n)])

    rotations = _best_standalone_rotations(case)
    sizes = np.array([_rotated_size(case, chiplet_id, rotations[chiplet_id]) for chiplet_id in ids], dtype=float)
    centers = _scale_points_to_outline(coords, sizes, case.outline_width, case.outline_height)
    raw_layout = _layout_from_centers(case, ids, centers, rotations)
    return _pack_by_order(case, raw_layout, rotations)


def _cgd_refine(case: FloorplanCase, layout: Layout, config: dict, seed: int) -> tuple[Layout, list[float]]:
    steps = int(config.get("cgd_steps", 500))
    if steps <= 0:
        return layout, []
    try:
        import torch
    except ImportError:
        return layout, []

    torch.manual_seed(seed)
    ids = [placement.chiplet_id for placement in layout.placements]
    rotations = {placement.chiplet_id: placement.rotation for placement in layout.placements}
    centers = np.array([_placement_center(case, placement) for placement in layout.placements], dtype=np.float64)
    xy = torch.tensor(centers, dtype=torch.float64, requires_grad=True)
    widths_np = np.array([_rotated_size(case, chiplet_id, rotations[chiplet_id])[0] for chiplet_id in ids], dtype=np.float64)
    heights_np = np.array([_rotated_size(case, chiplet_id, rotations[chiplet_id])[1] for chiplet_id in ids], dtype=np.float64)
    widths = torch.tensor(widths_np, dtype=torch.float64)
    heights = torch.tensor(heights_np, dtype=torch.float64)
    net_tensors = _net_tensors(case, ids, rotations)
    optimizer = torch.optim.Adam([xy], lr=float(config.get("learning_rate", 0.02)))

    density_weight = float(config.get("density_weight", 6000.0))
    outline_weight = float(config.get("outline_weight", 25000.0))
    wl_weight = float(config.get("wl_weight", 1.0))
    best_layout = layout
    best_score = _legal_hpwl_score(case, layout)
    curve: list[float] = []

    for step in range(steps):
        optimizer.zero_grad()
        overflow = _overlap_torch(xy, widths, heights)
        outline = _outline_torch(xy, widths, heights, case.outline_width, case.outline_height)
        wl = _smooth_hpwl_torch(xy, net_tensors)
        schedule = 1.0 + 2.0 * step / max(1, steps - 1)
        loss = wl_weight * wl + density_weight * schedule * overflow + outline_weight * outline
        loss.backward()
        optimizer.step()

        if step % max(1, int(config.get("report_every", 100))) == 0 or step == steps - 1:
            candidate = _layout_from_centers(case, ids, xy.detach().numpy(), rotations)
            legalized = _legalize_candidates(case, candidate, np.random.default_rng(seed + step), 8)
            score = _legal_hpwl_score(case, legalized)
            if score < best_score:
                best_layout = legalized
                best_score = score
            curve.append(best_score)
    return best_layout, curve


def _sequence_pair_polish(
    case: FloorplanCase,
    layout: Layout,
    objective: Callable[[Layout], CostResult],
    config: dict,
    rng: np.random.Generator,
) -> tuple[Layout, CostResult, list[float]]:
    iterations = int(config.get("legalize_iterations", 1200))
    cost = objective(layout)
    if iterations <= 0:
        return layout, cost, [cost.total]

    ids = [placement.chiplet_id for placement in layout.placements]
    rotations = {placement.chiplet_id: placement.rotation for placement in layout.placements}
    positive, negative = _sequence_pair_from_layout(case, layout)
    current = decode_sequence_pair(case, positive, negative, rotations)
    current_cost = objective(current)
    best = current if _is_legal(case, current) else layout
    best_cost = objective(best)
    curve = [best_cost.total]
    initial_temp = float(config.get("initial_anneal_temp", 0.25))
    final_temp = float(config.get("final_anneal_temp", 0.001))
    report_every = max(1, int(config.get("report_every", 100)))

    for step in range(iterations):
        frac = step / max(1, iterations - 1)
        temp = initial_temp * ((final_temp / initial_temp) ** frac)
        cand_positive = list(positive)
        cand_negative = list(negative)
        cand_rotations = dict(rotations)
        move = str(rng.choice(["swap_positive", "swap_negative", "swap_both", "rotate"], p=[0.30, 0.30, 0.28, 0.12]))
        if move == "rotate":
            chiplet_id = str(rng.choice(ids))
            cand_rotations[chiplet_id] = int(rng.choice(ROTATIONS))
        else:
            _swap_two(cand_positive if move in {"swap_positive", "swap_both"} else cand_negative, rng)
            if move == "swap_both":
                _swap_two(cand_negative, rng)
        candidate = decode_sequence_pair(case, cand_positive, cand_negative, cand_rotations)
        candidate_cost = objective(candidate)
        delta = candidate_cost.metrics["wirelength"] - current_cost.metrics["wirelength"]
        if delta <= 0.0 or rng.random() < exp(-delta / max(temp * max(1.0, current_cost.metrics["wirelength"]), 1e-12)):
            positive = cand_positive
            negative = cand_negative
            rotations = cand_rotations
            current = candidate
            current_cost = candidate_cost
            if _is_legal(case, current) and current_cost.metrics["wirelength"] < best_cost.metrics["wirelength"]:
                best = current
                best_cost = current_cost
        if step % report_every == 0 or step == iterations - 1:
            curve.append(best_cost.total)
    return best, best_cost, curve


def _legalize_candidates(case: FloorplanCase, layout: Layout, rng: np.random.Generator, count: int) -> Layout:
    rotations = {placement.chiplet_id: placement.rotation for placement in layout.placements}
    centers = {placement.chiplet_id: _placement_center(case, placement) for placement in layout.placements}
    candidates = [_pack_by_order(case, layout, rotations)]
    ordering_specs = [
        lambda item: (item[1][0] + item[1][1], item[1][0]),
        lambda item: (item[1][0] - item[1][1], item[1][0]),
        lambda item: (item[1][0], item[1][1]),
        lambda item: (item[1][1], item[1][0]),
    ]
    items = list(centers.items())
    for key in ordering_specs:
        ordered = [chiplet_id for chiplet_id, _ in sorted(items, key=key)]
        candidates.append(_decode_from_order(case, ordered, rotations))
    for _ in range(max(0, count - len(candidates))):
        jittered = sorted(items, key=lambda item: (item[1][0] + item[1][1] + rng.normal(0.0, 0.1), item[1][0]))
        candidates.append(_decode_from_order(case, [chiplet_id for chiplet_id, _ in jittered], rotations))
    legal = [candidate for candidate in candidates if _is_legal(case, candidate)]
    pool = legal or candidates
    return min(pool, key=lambda candidate: _legal_hpwl_score(case, candidate))


def _decode_from_order(case: FloorplanCase, ordered: list[str], rotations: dict[str, int]) -> Layout:
    reverse = list(reversed(ordered))
    options = [
        decode_sequence_pair(case, ordered, ordered, rotations),
        decode_sequence_pair(case, ordered, reverse, rotations),
        decode_sequence_pair(case, reverse, ordered, rotations),
    ]
    return min(options, key=lambda layout: _legal_hpwl_score(case, layout))


def _pack_by_order(case: FloorplanCase, layout: Layout, rotations: dict[str, int]) -> Layout:
    centers = {placement.chiplet_id: _placement_center(case, placement) for placement in layout.placements}
    ordered = [chiplet_id for chiplet_id, _ in sorted(centers.items(), key=lambda item: (item[1][1], item[1][0]))]
    return _shelf_pack(case, ordered, rotations)


def _shelf_pack(case: FloorplanCase, ordered: list[str], rotations: dict[str, int]) -> Layout:
    placements: list[Placement] = []
    x = 0.0
    y = 0.0
    row_height = 0.0
    spacing = 0.0
    for chiplet_id in ordered:
        width, height = _rotated_size(case, chiplet_id, rotations[chiplet_id])
        if x > 0 and x + width > case.outline_width:
            x = 0.0
            y += row_height + spacing
            row_height = 0.0
        if y + height > case.outline_height:
            return decode_sequence_pair(case, ordered, list(reversed(ordered)), rotations)
        placements.append(Placement(chiplet_id, x, y, rotations[chiplet_id]))
        x += width + spacing
        row_height = max(row_height, height)
    return Layout(tuple(placements))


def _best_standalone_rotations(case: FloorplanCase) -> dict[str, int]:
    rotations = {}
    for chiplet in case.chiplets:
        fits = []
        for rotation in ROTATIONS:
            width, height = _rotated_size(case, chiplet.id, rotation)
            if width <= case.outline_width and height <= case.outline_height:
                fits.append(rotation)
        rotations[chiplet.id] = fits[0] if fits else 0
    return rotations


def _scale_points_to_outline(points: np.ndarray, sizes: np.ndarray, outline_width: float, outline_height: float) -> np.ndarray:
    points = points.copy()
    for axis, outline, half_sizes in [(0, outline_width, sizes[:, 0] * 0.5), (1, outline_height, sizes[:, 1] * 0.5)]:
        lo = points[:, axis].min()
        hi = points[:, axis].max()
        if abs(hi - lo) < 1e-12:
            points[:, axis] = outline * 0.5
        else:
            span_lo = float(half_sizes.max())
            span_hi = float(outline - half_sizes.max())
            points[:, axis] = span_lo + (points[:, axis] - lo) / (hi - lo) * max(0.0, span_hi - span_lo)
    return points


def _choose_better_legal(
    case: FloorplanCase,
    left_layout: Layout,
    left_cost: CostResult,
    right_layout: Layout,
    right_cost: CostResult,
) -> tuple[Layout, CostResult]:
    left_legal = _is_legal(case, left_layout)
    right_legal = _is_legal(case, right_layout)
    if right_legal and (not left_legal or right_cost.metrics["wirelength"] < left_cost.metrics["wirelength"]):
        return right_layout, right_cost
    return left_layout, left_cost


def _is_legal(case: FloorplanCase, layout: Layout) -> bool:
    return total_overlap_penalty(case, layout) <= 1e-9 and total_outline_penalty(case, layout) <= 1e-9


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
        raw_offsets = net.pin_offsets or tuple((0.0, 0.0) for _ in net.chiplets)
        net_offsets = [_rotate_offset(raw[0], raw[1], rotations[chiplet_id]) for chiplet_id, raw in zip(net.chiplets, raw_offsets)]
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
    return hpwl(case, layout) + 1e9 * (total_overlap_penalty(case, layout) + total_outline_penalty(case, layout))


def _layout_from_centers(case: FloorplanCase, ids: list[str], centers: np.ndarray, rotations: dict[str, int]) -> Layout:
    placements = []
    for chiplet_id, (cx, cy) in zip(ids, centers):
        width, height = _rotated_size(case, chiplet_id, rotations[chiplet_id])
        placements.append(Placement(chiplet_id, float(cx - width * 0.5), float(cy - height * 0.5), rotations[chiplet_id]))
    return Layout(tuple(placements))


def _sequence_pair_from_layout(case: FloorplanCase, layout: Layout) -> tuple[list[str], list[str]]:
    centers = {placement.chiplet_id: _placement_center(case, placement) for placement in layout.placements}
    positive = sorted(centers, key=lambda chiplet_id: (centers[chiplet_id][0] + centers[chiplet_id][1], centers[chiplet_id][0]))
    negative = sorted(centers, key=lambda chiplet_id: (centers[chiplet_id][0] - centers[chiplet_id][1], centers[chiplet_id][0]))
    return positive, negative


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
