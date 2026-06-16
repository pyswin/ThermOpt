from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Callable

import numpy as np

from thermopt.layout.objects import FloorplanCase, Layout, Placement
from thermopt.objective.cost import CostResult


@dataclass(frozen=True)
class SAResult:
    best_layout: Layout
    best_cost: CostResult
    current_layout: Layout
    current_cost: CostResult
    best_curve: list[float]
    accepted_moves: int
    attempted_moves: int

    @property
    def accepted_ratio(self) -> float:
        return self.accepted_moves / max(1, self.attempted_moves)


def optimize(
    case: FloorplanCase,
    initial_layout: Layout,
    objective: Callable[[Layout], CostResult],
    config: dict,
    seed: int,
) -> SAResult:
    rng = np.random.default_rng(seed)
    iterations = int(config.get("iterations", 1000))
    initial_temp = float(config.get("initial_anneal_temp", 1.0))
    final_temp = float(config.get("final_anneal_temp", 0.01))
    move_scale = float(config.get("move_scale", 10.0))
    report_every = max(1, int(config.get("report_every", 10)))

    current_layout = initial_layout
    current_cost = objective(current_layout)
    best_layout = current_layout
    best_cost = current_cost
    best_curve = [best_cost.total]
    accepted = 0

    for step in range(iterations):
        frac = step / max(1, iterations - 1)
        anneal_temp = initial_temp * ((final_temp / initial_temp) ** frac)
        candidate = propose_move(case, current_layout, rng, move_scale)
        candidate_cost = objective(candidate)
        delta = candidate_cost.total - current_cost.total
        if delta <= 0.0 or rng.random() < exp(-delta / max(anneal_temp, 1e-12)):
            current_layout = candidate
            current_cost = candidate_cost
            accepted += 1
            if current_cost.total < best_cost.total:
                best_layout = current_layout
                best_cost = current_cost
        if step % report_every == 0 or step == iterations - 1:
            best_curve.append(best_cost.total)

    return SAResult(
        best_layout=best_layout,
        best_cost=best_cost,
        current_layout=current_layout,
        current_cost=current_cost,
        best_curve=best_curve,
        accepted_moves=accepted,
        attempted_moves=iterations,
    )


def propose_move(case: FloorplanCase, layout: Layout, rng: np.random.Generator, move_scale: float) -> Layout:
    move = str(rng.choice(["translate", "swap", "rotate", "perturb"], p=[0.45, 0.2, 0.15, 0.2]))
    return propose_named_move(case, layout, rng, move_scale, move)


def propose_named_move(
    case: FloorplanCase,
    layout: Layout,
    rng: np.random.Generator,
    move_scale: float,
    move: str,
) -> Layout:
    if move not in {"translate", "swap", "rotate", "perturb"}:
        raise ValueError(f"unknown move type: {move}")
    placements = list(layout.placements)
    idx = int(rng.integers(0, len(placements)))

    if move == "swap" and len(placements) >= 2:
        j = int(rng.integers(0, len(placements) - 1))
        if j >= idx:
            j += 1
        a, b = placements[idx], placements[j]
        placements[idx] = Placement(a.chiplet_id, b.x, b.y, a.rotation)
        placements[j] = Placement(b.chiplet_id, a.x, a.y, b.rotation)
        return Layout(tuple(placements))

    placement = placements[idx]
    chiplet = case.chiplet_by_id[placement.chiplet_id]

    if move == "rotate":
        placements[idx] = placement.rotated()
        return Layout(tuple(placements))

    scale = move_scale if move == "translate" else move_scale * 0.35
    dx, dy = rng.normal(0.0, scale, size=2)
    width, height = placement.rotated_size(chiplet)
    # Keep most proposals near the outline while still allowing penalties to guide recovery.
    margin = max(case.outline_width, case.outline_height) * 0.1
    x = float(np.clip(placement.x + dx, -margin, case.outline_width - width + margin))
    y = float(np.clip(placement.y + dy, -margin, case.outline_height - height + margin))
    placements[idx] = placement.moved(x=x, y=y)
    return Layout(tuple(placements))
