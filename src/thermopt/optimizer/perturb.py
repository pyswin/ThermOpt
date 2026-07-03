from __future__ import annotations

"""
Position-only perturbation operators (swap / move / rotate) used to build a
diverse dataset of legal-ish layouts around a set of MILP seed positions.
No thermal simulation involved — everything here is pure geometry.
"""

from dataclasses import dataclass

import numpy as np

from thermopt.layout.objects import FloorplanCase, Layout, Placement
from thermopt.layout.geometry import total_outline_penalty, total_overlap_penalty
from thermopt.optimizer.atplace import _legalize_by_sequence_pair

MOVE_SMALL = "move_small"
MOVE_MORE = "move_more"
SWAP = "swap"
ROTATE = "rotate"


@dataclass(frozen=True)
class PerturbResult:
    layout: Layout
    move_type: str
    touched: tuple[str, ...]
    overlap: float
    outline: float
    legal: bool


def legalize(case: FloorplanCase, layout: Layout) -> Layout:
    return _legalize_by_sequence_pair(case, layout)


def rectangular_ids(case: FloorplanCase) -> list[str]:
    return [c.id for c in case.chiplets if abs(c.width - c.height) > 1e-9]


def power_weights(case: FloorplanCase, ids: list[str]) -> np.ndarray:
    """Sampling weights proportional to power (the only per-chiplet heat proxy
    available without running a thermal simulation)."""
    powers = np.array([max(case.chiplet_by_id[i].power, 0.0) for i in ids], dtype=float)
    # small floor so zero-power chiplets can still occasionally be picked
    weights = powers + 0.05 * (powers.max() if powers.max() > 0 else 1.0)
    return weights / weights.sum()


def _weighted_pick(rng: np.random.Generator, ids: list[str], weights: np.ndarray, k: int) -> list[str]:
    k = min(k, len(ids))
    chosen = rng.choice(len(ids), size=k, replace=False, p=weights)
    return [ids[i] for i in chosen]


def swap_move(case: FloorplanCase, layout: Layout, rng: np.random.Generator) -> tuple[Layout, tuple[str, ...]]:
    ids = [p.chiplet_id for p in layout.placements]
    weights = power_weights(case, ids)
    a_id, b_id = _weighted_pick(rng, ids, weights, 2)
    by_id = layout.by_id
    a, b = by_id[a_id], by_id[b_id]
    new_layout = layout.replace_many(
        [Placement(a_id, b.x, b.y, a.rotation), Placement(b_id, a.x, a.y, b.rotation)]
    )
    return new_layout, (a_id, b_id)


def translate_move(
    case: FloorplanCase,
    layout: Layout,
    rng: np.random.Generator,
    k_min: int,
    k_max: int,
    max_move_frac: float,
) -> tuple[Layout, tuple[str, ...]]:
    ids = [p.chiplet_id for p in layout.placements]
    n = len(ids)
    k_max = max(k_min, min(k_max, n))
    k = int(rng.integers(k_min, k_max + 1))
    weights = power_weights(case, ids)
    touched = _weighted_pick(rng, ids, weights, k)

    max_move = max_move_frac * min(case.outline_width, case.outline_height)
    by_id = layout.by_id
    updates = []
    for chiplet_id in touched:
        placement = by_id[chiplet_id]
        chiplet = case.chiplet_by_id[chiplet_id]
        width, height = placement.rotated_size(chiplet)
        dx, dy = rng.uniform(-max_move, max_move, size=2)
        x = float(np.clip(placement.x + dx, -0.1 * max_move, case.outline_width - width + 0.1 * max_move))
        y = float(np.clip(placement.y + dy, -0.1 * max_move, case.outline_height - height + 0.1 * max_move))
        updates.append(placement.moved(x=x, y=y))
    return layout.replace_many(updates), tuple(touched)


def rotate_move(case: FloorplanCase, layout: Layout, rng: np.random.Generator) -> tuple[Layout, tuple[str, ...]] | None:
    candidates = rectangular_ids(case)
    if not candidates:
        return None
    weights = power_weights(case, candidates)
    chiplet_id = _weighted_pick(rng, candidates, weights, 1)[0]
    placement = layout.by_id[chiplet_id]
    steps = int(rng.integers(1, 4))  # rotate by 90/180/270
    rotation = placement
    for _ in range(steps):
        rotation = rotation.rotated()
    return layout.replace_placement(rotation), (chiplet_id,)


def move_weights(allow_rotate: bool) -> tuple[list[str], list[float]]:
    moves = [SWAP, MOVE_SMALL, MOVE_MORE] + ([ROTATE] if allow_rotate else [])
    base = {SWAP: 0.30, MOVE_SMALL: 0.25, MOVE_MORE: 0.25, ROTATE: 0.20}
    weights = [base[m] for m in moves]
    total = sum(weights)
    return moves, [w / total for w in weights]


def generate_one(
    case: FloorplanCase,
    layout: Layout,
    rng: np.random.Generator,
    small_k_range: tuple[int, int],
    more_k_range: tuple[int, int],
    max_move_frac: float = 0.12,
) -> PerturbResult:
    n = len(case.chiplets)
    allow_rotate = len(rectangular_ids(case)) > 0
    moves, probs = move_weights(allow_rotate)
    move_type = str(rng.choice(moves, p=probs))

    if move_type == SWAP:
        candidate, touched = swap_move(case, layout, rng)
    elif move_type == MOVE_SMALL:
        candidate, touched = translate_move(case, layout, rng, *small_k_range, max_move_frac)
    elif move_type == MOVE_MORE:
        candidate, touched = translate_move(case, layout, rng, *more_k_range, max_move_frac)
    else:  # ROTATE
        rotated = rotate_move(case, layout, rng)
        if rotated is None:
            candidate, touched = translate_move(case, layout, rng, *small_k_range, max_move_frac)
            move_type = MOVE_SMALL
        else:
            candidate, touched = rotated

    legal_layout = legalize(case, candidate)
    overlap = total_overlap_penalty(case, legal_layout)
    outline = total_outline_penalty(case, legal_layout)
    return PerturbResult(
        layout=legal_layout,
        move_type=move_type,
        touched=touched,
        overlap=overlap,
        outline=outline,
        legal=overlap <= 1e-9 and outline <= 1e-9,
    )
