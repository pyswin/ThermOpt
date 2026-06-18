from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from thermopt.layout.geometry import hpwl, total_outline_penalty, total_overlap_penalty
from thermopt.layout.objects import FloorplanCase, Layout, Placement
from thermopt.objective.cost import CostResult
from thermopt.optimizer.sequence_pair import decode_sequence_pair


@dataclass(frozen=True)
class NesterovResult:
    best_layout: Layout
    best_cost: CostResult
    best_curve: list[float]
    phases: list[dict[str, float | str | bool]]
    steps: int


def optimize(
    case: FloorplanCase,
    initial_layout: Layout,
    objective: Callable[[Layout], CostResult],
    config: dict,
    seed: int,
) -> NesterovResult:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("nesterov optimizer requires torch") from exc

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    ids = [chiplet.id for chiplet in case.chiplets]
    rotations = _initial_rotations(case, initial_layout, config)
    start_layout = _initial_layout(case, initial_layout, rotations, config)
    start_cost = objective(start_layout)
    phases: list[dict[str, float | str | bool]] = []
    _record(phases, "initial", start_layout, start_cost, case)

    centers = np.array([_placement_center(case, start_layout.by_id[chiplet_id]) for chiplet_id in ids], dtype=np.float64)
    sizes = np.array([_rotated_size(case, chiplet_id, rotations[chiplet_id]) for chiplet_id in ids], dtype=np.float64)
    xy = torch.tensor(centers, dtype=torch.float64)
    velocity = torch.zeros_like(xy)
    widths = torch.tensor(sizes[:, 0], dtype=torch.float64)
    heights = torch.tensor(sizes[:, 1], dtype=torch.float64)
    net_tensors = _net_tensors(case, ids, rotations)

    steps = int(config.get("steps", 800))
    learning_rate = float(config.get("learning_rate", 0.02))
    momentum = float(config.get("momentum", 0.92))
    wl_weight = float(config.get("wl_weight", 1.0))
    density_weight = float(config.get("density_weight", 3000.0))
    outline_weight = float(config.get("outline_weight", 20000.0))
    overlap_weight = float(config.get("overlap_weight", 1000.0))
    report_every = max(1, int(config.get("report_every", 100)))

    best_layout = _legalize(case, start_layout, rng, int(config.get("legalize_candidates", 32)))
    best_cost = objective(best_layout)
    best_curve = [best_cost.total]

    for step in range(steps):
        lookahead = (xy + momentum * velocity).detach().clone().requires_grad_(True)
        wl = _weighted_average_hpwl(lookahead, net_tensors, gamma=float(config.get("wa_gamma", 0.35)))
        density = _density_overflow(lookahead, widths, heights, case, config)
        overlap = _overlap_penalty(lookahead, widths, heights)
        outline = _outline_penalty(lookahead, widths, heights, case.outline_width, case.outline_height)
        schedule = 1.0 + float(config.get("density_ramp", 2.0)) * step / max(1, steps - 1)
        loss = wl_weight * wl + density_weight * schedule * density + overlap_weight * overlap + outline_weight * outline
        loss.backward()
        grad = lookahead.grad.detach()
        grad_norm = torch.linalg.vector_norm(grad).clamp_min(1e-9)
        step_size = learning_rate / grad_norm.sqrt()
        velocity = momentum * velocity - step_size * grad
        xy = (xy + velocity).detach()
        xy[:, 0].clamp_(0.0, case.outline_width)
        xy[:, 1].clamp_(0.0, case.outline_height)

        if step % report_every == 0 or step == steps - 1:
            candidate = _layout_from_centers(case, ids, xy.numpy(), rotations)
            legalized = _legalize(case, candidate, rng, int(config.get("legalize_candidates", 32)))
            candidate_cost = objective(legalized)
            if _is_legal(case, legalized) and candidate_cost.metrics["wirelength"] < best_cost.metrics["wirelength"]:
                best_layout = legalized
                best_cost = candidate_cost
            best_curve.append(best_cost.total)

    _record(phases, "nesterov_global", _layout_from_centers(case, ids, xy.numpy(), rotations), objective(_layout_from_centers(case, ids, xy.numpy(), rotations)), case)
    _record(phases, "legalization", best_layout, best_cost, case)
    return NesterovResult(best_layout=best_layout, best_cost=best_cost, best_curve=best_curve, phases=phases, steps=steps)


def _initial_rotations(case: FloorplanCase, initial_layout: Layout, config: dict) -> dict[str, int]:
    mode = str(config.get("rotation_mode", "initial")).lower()
    if mode == "zero":
        return {chiplet.id: 0 for chiplet in case.chiplets}
    return {placement.chiplet_id: placement.rotation for placement in initial_layout.placements}


def _initial_layout(case: FloorplanCase, initial_layout: Layout, rotations: dict[str, int], config: dict) -> Layout:
    if str(config.get("initialization", "spectral")).lower() != "spectral":
        return initial_layout
    ids = [chiplet.id for chiplet in case.chiplets]
    index = {chiplet_id: i for i, chiplet_id in enumerate(ids)}
    weights = np.zeros((len(ids), len(ids)), dtype=float)
    for net in case.nets:
        unique = list(dict.fromkeys(net.chiplets))
        for a in range(len(unique)):
            for b in range(a + 1, len(unique)):
                i = index[unique[a]]
                j = index[unique[b]]
                weights[i, j] += 1.0
                weights[j, i] += 1.0
    laplacian = np.diag(weights.sum(axis=1)) - weights
    try:
        _, vectors = np.linalg.eigh(laplacian)
        points = vectors[:, 1:3] if len(ids) >= 3 else np.column_stack([np.arange(len(ids)), np.zeros(len(ids))])
    except np.linalg.LinAlgError:
        points = np.column_stack([np.arange(len(ids)), np.zeros(len(ids))])
    if points.shape[1] < 2:
        points = np.column_stack([points[:, 0], np.zeros(len(ids))])
    sizes = np.array([_rotated_size(case, chiplet_id, rotations[chiplet_id]) for chiplet_id in ids], dtype=float)
    centers = _scale_points(points, sizes, case.outline_width, case.outline_height)
    return _layout_from_centers(case, ids, centers, rotations)


def _weighted_average_hpwl(xy, net_tensors, gamma: float):
    import torch

    indices, offsets, mask = net_tensors
    pins = xy[indices] + offsets
    x = pins[:, :, 0]
    y = pins[:, :, 1]
    invalid_low = (1.0 - mask) * -1e6
    invalid_high = (1.0 - mask) * 1e6

    def wa(values):
        hi = torch.softmax((values + invalid_low) / gamma, dim=1)
        lo = torch.softmax((-values - invalid_high) / gamma, dim=1)
        return (hi * values * mask).sum(dim=1) - (lo * values * mask).sum(dim=1)

    return (wa(x) + wa(y)).sum()


def _density_overflow(xy, widths, heights, case: FloorplanCase, config: dict):
    import torch

    bins_x = int(config.get("density_bins_x", 16))
    bins_y = int(config.get("density_bins_y", 16))
    target = float(config.get("target_density", min(0.95, case.total_chiplet_area / max(case.outline_width * case.outline_height, 1e-9) * 1.15)))
    xgrid = (torch.arange(bins_x, dtype=torch.float64) + 0.5) / bins_x * case.outline_width
    ygrid = (torch.arange(bins_y, dtype=torch.float64) + 0.5) / bins_y * case.outline_height
    gx, gy = torch.meshgrid(xgrid, ygrid, indexing="ij")
    sigma_x = case.outline_width / bins_x * float(config.get("density_sigma", 1.5))
    sigma_y = case.outline_height / bins_y * float(config.get("density_sigma", 1.5))
    area = (widths * heights).view(-1, 1, 1)
    dx = (gx.view(1, bins_x, bins_y) - xy[:, 0].view(-1, 1, 1)) / max(sigma_x, 1e-9)
    dy = (gy.view(1, bins_x, bins_y) - xy[:, 1].view(-1, 1, 1)) / max(sigma_y, 1e-9)
    density = (area * torch.exp(-0.5 * (dx.square() + dy.square()))).sum(dim=0)
    bin_area = case.outline_width / bins_x * case.outline_height / bins_y
    capacity = target * bin_area
    return torch.relu(density - capacity).square().mean() / max(bin_area * bin_area, 1e-9)


def _overlap_penalty(xy, widths, heights):
    import torch

    penalty = xy.new_tensor(0.0)
    for i in range(len(widths)):
        dx = torch.relu((widths[i] + widths[i + 1 :]) * 0.5 - torch.abs(xy[i, 0] - xy[i + 1 :, 0]))
        dy = torch.relu((heights[i] + heights[i + 1 :]) * 0.5 - torch.abs(xy[i, 1] - xy[i + 1 :, 1]))
        penalty = penalty + (dx * dy).sum()
    return penalty


def _outline_penalty(xy, widths, heights, outline_width: float, outline_height: float):
    import torch

    return (
        torch.relu(widths * 0.5 - xy[:, 0]).square()
        + torch.relu(xy[:, 0] + widths * 0.5 - outline_width).square()
        + torch.relu(heights * 0.5 - xy[:, 1]).square()
        + torch.relu(xy[:, 1] + heights * 0.5 - outline_height).square()
    ).sum()


def _legalize(case: FloorplanCase, layout: Layout, rng: np.random.Generator, candidates: int) -> Layout:
    rotations = {placement.chiplet_id: placement.rotation for placement in layout.placements}
    centers = {placement.chiplet_id: _placement_center(case, placement) for placement in layout.placements}
    items = list(centers.items())
    layouts = []
    keys = [
        lambda item: (item[1][0] + item[1][1], item[1][0]),
        lambda item: (item[1][0] - item[1][1], item[1][0]),
        lambda item: (item[1][0], item[1][1]),
        lambda item: (item[1][1], item[1][0]),
    ]
    for key in keys:
        order = [chiplet_id for chiplet_id, _ in sorted(items, key=key)]
        layouts.extend(_decode_orders(case, order, rotations))
    for _ in range(max(0, candidates - len(layouts))):
        order = [
            chiplet_id
            for chiplet_id, _ in sorted(
                items,
                key=lambda item: (item[1][0] + item[1][1] + rng.normal(0.0, 0.1), item[1][0]),
            )
        ]
        layouts.extend(_decode_orders(case, order, rotations)[:1])
    legal = [item for item in layouts if _is_legal(case, item)]
    return min(legal or layouts, key=lambda item: _legal_score(case, item))


def _decode_orders(case: FloorplanCase, order: list[str], rotations: dict[str, int]) -> list[Layout]:
    rev = list(reversed(order))
    return [
        decode_sequence_pair(case, order, order, rotations),
        decode_sequence_pair(case, order, rev, rotations),
        decode_sequence_pair(case, rev, order, rotations),
    ]


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


def _is_legal(case: FloorplanCase, layout: Layout) -> bool:
    return total_overlap_penalty(case, layout) <= 1e-9 and total_outline_penalty(case, layout) <= 1e-9


def _legal_score(case: FloorplanCase, layout: Layout) -> float:
    return hpwl(case, layout) + 1e9 * (total_overlap_penalty(case, layout) + total_outline_penalty(case, layout))


def _scale_points(points: np.ndarray, sizes: np.ndarray, outline_width: float, outline_height: float) -> np.ndarray:
    points = points.copy()
    for axis, outline, half_sizes in [(0, outline_width, sizes[:, 0] * 0.5), (1, outline_height, sizes[:, 1] * 0.5)]:
        lo = points[:, axis].min()
        hi = points[:, axis].max()
        if abs(hi - lo) < 1e-12:
            points[:, axis] = outline * 0.5
        else:
            low = float(half_sizes.max())
            high = float(outline - half_sizes.max())
            points[:, axis] = low + (points[:, axis] - lo) / (hi - lo) * max(0.0, high - low)
    return points


def _layout_from_centers(case: FloorplanCase, ids: list[str], centers: np.ndarray, rotations: dict[str, int]) -> Layout:
    placements = []
    for chiplet_id, (cx, cy) in zip(ids, centers):
        width, height = _rotated_size(case, chiplet_id, rotations[chiplet_id])
        placements.append(Placement(chiplet_id, float(cx - width * 0.5), float(cy - height * 0.5), rotations[chiplet_id]))
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
