from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Callable

import numpy as np

from thermopt.layout.objects import FloorplanCase, Layout, Placement
from thermopt.objective.cost import CostResult


@dataclass(frozen=True)
class SequencePairResult:
    best_layout: Layout
    best_cost: CostResult
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
) -> SequencePairResult:
    rng = np.random.default_rng(seed)
    iterations = int(config.get("iterations", 2000))
    initial_temp = float(config.get("initial_anneal_temp", 1.0))
    final_temp = float(config.get("final_anneal_temp", 0.01))
    report_every = max(1, int(config.get("report_every", 25)))

    ids = [placement.chiplet_id for placement in initial_layout.placements]
    positive = list(ids)
    negative = list(ids)
    rotations = {placement.chiplet_id: placement.rotation for placement in initial_layout.placements}

    current_layout = decode_sequence_pair(case, positive, negative, rotations)
    current_cost = objective(current_layout)
    best_layout = current_layout
    best_cost = current_cost
    best_curve = [best_cost.total]
    accepted = 0

    for step in range(iterations):
        frac = step / max(1, iterations - 1)
        anneal_temp = initial_temp * ((final_temp / initial_temp) ** frac)
        cand_positive = list(positive)
        cand_negative = list(negative)
        cand_rotations = dict(rotations)

        move = str(rng.choice(["swap_positive", "swap_negative", "swap_both", "rotate"], p=[0.35, 0.35, 0.2, 0.1]))
        if move == "rotate":
            chiplet_id = str(rng.choice(ids))
            cand_rotations[chiplet_id] = (cand_rotations[chiplet_id] + 90) % 360
        else:
            _swap_two(cand_positive if move in {"swap_positive", "swap_both"} else cand_negative, rng)
            if move == "swap_both":
                _swap_two(cand_negative, rng)

        candidate_layout = decode_sequence_pair(case, cand_positive, cand_negative, cand_rotations)
        candidate_cost = objective(candidate_layout)
        delta = candidate_cost.total - current_cost.total
        if delta <= 0.0 or rng.random() < exp(-delta / max(anneal_temp, 1e-12)):
            positive = cand_positive
            negative = cand_negative
            rotations = cand_rotations
            current_layout = candidate_layout
            current_cost = candidate_cost
            accepted += 1
            if current_cost.total < best_cost.total:
                best_layout = current_layout
                best_cost = current_cost

        if step % report_every == 0 or step == iterations - 1:
            best_curve.append(best_cost.total)

    return SequencePairResult(best_layout, best_cost, best_curve, accepted, iterations)


def _swap_two(values: list[str], rng: np.random.Generator) -> None:
    if len(values) < 2:
        return
    i, j = rng.choice(len(values), size=2, replace=False)
    values[int(i)], values[int(j)] = values[int(j)], values[int(i)]


def decode_sequence_pair(
    case: FloorplanCase,
    positive: list[str],
    negative: list[str],
    rotations: dict[str, int],
) -> Layout:
    pos_index = {chiplet_id: index for index, chiplet_id in enumerate(positive)}
    neg_index = {chiplet_id: index for index, chiplet_id in enumerate(negative)}
    ids = list(positive)
    sizes: dict[str, tuple[float, float]] = {}
    for chiplet_id in ids:
        chiplet = case.chiplet_by_id[chiplet_id]
        rotation = rotations.get(chiplet_id, 0)
        sizes[chiplet_id] = (chiplet.height, chiplet.width) if rotation % 180 == 90 else (chiplet.width, chiplet.height)

    x = {chiplet_id: 0.0 for chiplet_id in ids}
    y = {chiplet_id: 0.0 for chiplet_id in ids}
    for source in ids:
        for target in ids:
            if source == target:
                continue
            if pos_index[source] < pos_index[target]:
                if neg_index[source] < neg_index[target]:
                    x[target] = max(x[target], x[source] + sizes[source][0])
                else:
                    y[target] = max(y[target], y[source] + sizes[source][1])

    return Layout(
        tuple(
            Placement(chiplet_id=chiplet_id, x=x[chiplet_id], y=y[chiplet_id], rotation=rotations.get(chiplet_id, 0))
            for chiplet_id in ids
        )
    )
