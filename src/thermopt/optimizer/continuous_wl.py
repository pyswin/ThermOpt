from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from thermopt.layout.objects import FloorplanCase, Layout, Placement
from thermopt.objective.cost import CostResult


@dataclass(frozen=True)
class ContinuousWLResult:
    best_layout: Layout
    best_cost: CostResult
    best_curve: list[float]
    steps: int


def optimize(
    case: FloorplanCase,
    initial_layout: Layout,
    objective: Callable[[Layout], CostResult],
    config: dict,
    seed: int,
) -> ContinuousWLResult:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("continuous_wl optimizer requires torch") from exc

    rng = np.random.default_rng(seed)
    steps = int(config.get("steps", 3000))
    restarts = max(1, int(config.get("restarts", 4)))
    learning_rate = float(config.get("learning_rate", 0.04))
    overlap_weight = float(config.get("overlap_weight", 20000.0))
    outline_weight = float(config.get("outline_weight", 100000.0))
    report_every = max(1, int(config.get("report_every", 100)))

    best_layout = initial_layout
    best_cost = objective(initial_layout)
    best_curve = [best_cost.total]

    ids = [chiplet.id for chiplet in case.chiplets]
    index = {chiplet_id: i for i, chiplet_id in enumerate(ids)}
    widths_np = np.array([chiplet.width for chiplet in case.chiplets], dtype=np.float64)
    heights_np = np.array([chiplet.height for chiplet in case.chiplets], dtype=np.float64)
    net_left, net_right, off_left, off_right = _two_pin_net_arrays(case, index)

    device = torch.device("cpu")
    widths = torch.tensor(widths_np, dtype=torch.float64, device=device)
    heights = torch.tensor(heights_np, dtype=torch.float64, device=device)
    left = torch.tensor(net_left, dtype=torch.long, device=device)
    right = torch.tensor(net_right, dtype=torch.long, device=device)
    left_offsets = torch.tensor(off_left, dtype=torch.float64, device=device)
    right_offsets = torch.tensor(off_right, dtype=torch.float64, device=device)

    initial_centers = np.array(
        [
            [
                initial_layout.by_id[chiplet_id].x + case.chiplet_by_id[chiplet_id].width * 0.5,
                initial_layout.by_id[chiplet_id].y + case.chiplet_by_id[chiplet_id].height * 0.5,
            ]
            for chiplet_id in ids
        ],
        dtype=np.float64,
    )

    for restart in range(restarts):
        centers = initial_centers.copy()
        if restart > 0:
            centers[:, 0] = rng.uniform(widths_np * 0.5, case.outline_width - widths_np * 0.5)
            centers[:, 1] = rng.uniform(heights_np * 0.5, case.outline_height - heights_np * 0.5)

        xy = torch.tensor(centers, dtype=torch.float64, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([xy], lr=learning_rate)

        for step in range(steps):
            optimizer.zero_grad()
            pins_left = xy[left] + left_offsets
            pins_right = xy[right] + right_offsets
            diff = pins_left - pins_right
            wl = torch.sqrt(diff[:, 0] ** 2 + 1e-6).sum() + torch.sqrt(diff[:, 1] ** 2 + 1e-6).sum()
            outline = _outline_penalty_torch(xy, widths, heights, case.outline_width, case.outline_height)
            overlap = _overlap_penalty_torch(xy, widths, heights)
            loss = wl + outline_weight * outline + overlap_weight * overlap
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                xy[:, 0].clamp_(0.0, case.outline_width)
                xy[:, 1].clamp_(0.0, case.outline_height)

            if step % report_every == 0 or step == steps - 1:
                layout = _layout_from_centers(case, ids, xy.detach().cpu().numpy())
                cost = objective(layout)
                best_curve.append(min(best_curve[-1], cost.total))
                if cost.total < best_cost.total:
                    best_layout = layout
                    best_cost = cost

    return ContinuousWLResult(best_layout=best_layout, best_cost=best_cost, best_curve=best_curve, steps=steps * restarts)


def _two_pin_net_arrays(
    case: FloorplanCase,
    index: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left: list[int] = []
    right: list[int] = []
    left_offsets: list[tuple[float, float]] = []
    right_offsets: list[tuple[float, float]] = []
    for net in case.nets:
        if len(net.chiplets) != 2 or len(set(net.chiplets)) < 2:
            continue
        offsets = net.pin_offsets or ((0.0, 0.0), (0.0, 0.0))
        left.append(index[net.chiplets[0]])
        right.append(index[net.chiplets[1]])
        left_offsets.append(offsets[0])
        right_offsets.append(offsets[1])
    return (
        np.array(left, dtype=np.int64),
        np.array(right, dtype=np.int64),
        np.array(left_offsets, dtype=np.float64),
        np.array(right_offsets, dtype=np.float64),
    )


def _outline_penalty_torch(xy, widths, heights, outline_width: float, outline_height: float):
    import torch

    return (
        torch.relu(widths * 0.5 - xy[:, 0]).square()
        + torch.relu(xy[:, 0] + widths * 0.5 - outline_width).square()
        + torch.relu(heights * 0.5 - xy[:, 1]).square()
        + torch.relu(xy[:, 1] + heights * 0.5 - outline_height).square()
    ).sum()


def _overlap_penalty_torch(xy, widths, heights):
    import torch

    overlap = xy.new_tensor(0.0)
    for i in range(len(widths)):
        dx = torch.relu((widths[i] + widths[i + 1 :]) * 0.5 - torch.abs(xy[i, 0] - xy[i + 1 :, 0]))
        dy = torch.relu((heights[i] + heights[i + 1 :]) * 0.5 - torch.abs(xy[i, 1] - xy[i + 1 :, 1]))
        overlap = overlap + (dx * dy).sum()
    return overlap


def _layout_from_centers(case: FloorplanCase, ids: list[str], centers: np.ndarray) -> Layout:
    placements = []
    for chiplet_id, (cx, cy) in zip(ids, centers):
        chiplet = case.chiplet_by_id[chiplet_id]
        placements.append(Placement(chiplet_id, float(cx - chiplet.width * 0.5), float(cy - chiplet.height * 0.5)))
    return Layout(tuple(placements))
