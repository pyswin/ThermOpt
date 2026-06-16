from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from thermopt.layout.objects import FloorplanCase, Layout, Placement
from thermopt.objective.cost import CostResult
from thermopt.optimizer.simulated_annealing import propose_move


@dataclass(frozen=True)
class GAResult:
    best_layout: Layout
    best_cost: CostResult
    best_curve: list[float]
    generations: int
    population_size: int


def optimize(
    case: FloorplanCase,
    initial_layout: Layout,
    objective: Callable[[Layout], CostResult],
    config: dict,
    seed: int,
) -> GAResult:
    rng = np.random.default_rng(seed)
    population_size = max(2, int(config.get("population_size", 24)))
    generations = max(1, int(config.get("generations", 40)))
    elite_count = min(population_size, max(1, int(config.get("elite_count", 2))))
    mutation_rate = float(config.get("mutation_rate", 0.35))
    move_scale = float(config.get("move_scale", 10.0))
    tournament_size = min(population_size, max(2, int(config.get("tournament_size", 3))))

    population = _initial_population(case, initial_layout, population_size, rng, move_scale)
    scored = _score_population(population, objective)
    best_layout, best_cost = scored[0]
    best_curve = [best_cost.total]

    for _ in range(generations):
        next_population = [layout for layout, _ in scored[:elite_count]]
        while len(next_population) < population_size:
            parent_a = _tournament_select(scored, tournament_size, rng)
            parent_b = _tournament_select(scored, tournament_size, rng)
            child = _crossover(parent_a, parent_b, rng)
            if rng.random() < mutation_rate:
                child = propose_move(case, child, rng, move_scale)
            next_population.append(child)

        scored = _score_population(next_population, objective)
        if scored[0][1].total < best_cost.total:
            best_layout, best_cost = scored[0]
        best_curve.append(best_cost.total)

    return GAResult(
        best_layout=best_layout,
        best_cost=best_cost,
        best_curve=best_curve,
        generations=generations,
        population_size=population_size,
    )


def _initial_population(
    case: FloorplanCase,
    initial_layout: Layout,
    population_size: int,
    rng: np.random.Generator,
    move_scale: float,
) -> list[Layout]:
    population = [initial_layout]
    while len(population) < population_size:
        layout = initial_layout
        for _ in range(int(rng.integers(1, 5))):
            layout = propose_move(case, layout, rng, move_scale)
        population.append(layout)
    return population


def _score_population(
    population: list[Layout],
    objective: Callable[[Layout], CostResult],
) -> list[tuple[Layout, CostResult]]:
    scored = [(layout, objective(layout)) for layout in population]
    return sorted(scored, key=lambda item: item[1].total)


def _tournament_select(
    scored: list[tuple[Layout, CostResult]],
    tournament_size: int,
    rng: np.random.Generator,
) -> Layout:
    indices = rng.choice(len(scored), size=tournament_size, replace=False)
    candidates = [scored[int(index)] for index in indices]
    return min(candidates, key=lambda item: item[1].total)[0]


def _crossover(parent_a: Layout, parent_b: Layout, rng: np.random.Generator) -> Layout:
    b_by_id = parent_b.by_id
    child: list[Placement] = []
    for placement_a in parent_a.placements:
        source = placement_a if rng.random() < 0.5 else b_by_id[placement_a.chiplet_id]
        child.append(Placement(source.chiplet_id, source.x, source.y, source.rotation))
    return Layout(tuple(child))
