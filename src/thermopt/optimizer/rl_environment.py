from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from thermopt.layout.objects import FloorplanCase, Layout
from thermopt.objective.cost import CostResult
from thermopt.optimizer.simulated_annealing import propose_move


@dataclass(frozen=True)
class StepResult:
    observation: dict
    reward: float
    done: bool
    info: dict


class ThermalFloorplanEnv:
    """Small dependency-free RL-style environment for floorplanning experiments."""

    action_names = ("translate", "swap", "rotate", "perturb")

    def __init__(
        self,
        case: FloorplanCase,
        initial_layout: Layout,
        objective: Callable[[Layout], CostResult],
        max_steps: int = 100,
        move_scale: float = 10.0,
        seed: int = 0,
    ) -> None:
        self.case = case
        self.initial_layout = initial_layout
        self.objective = objective
        self.max_steps = max_steps
        self.move_scale = move_scale
        self.rng = np.random.default_rng(seed)
        self.layout = initial_layout
        self.cost = objective(initial_layout)
        self.steps = 0

    def reset(self, seed: int | None = None) -> dict:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.layout = self.initial_layout
        self.cost = self.objective(self.layout)
        self.steps = 0
        return self._observation()

    def step(self, action: int) -> StepResult:
        if action < 0 or action >= len(self.action_names):
            raise ValueError(f"action must be in [0, {len(self.action_names) - 1}]")
        previous_cost = self.cost.total
        self.layout = propose_move(self.case, self.layout, self.rng, self.move_scale)
        self.cost = self.objective(self.layout)
        self.steps += 1
        reward = previous_cost - self.cost.total
        done = self.steps >= self.max_steps
        return StepResult(
            observation=self._observation(),
            reward=float(reward),
            done=done,
            info={
                "action": self.action_names[action],
                "cost": self.cost.total,
                "metrics": self.cost.metrics,
            },
        )

    def _observation(self) -> dict:
        return {
            "step": self.steps,
            "cost": self.cost.total,
            "placements": [
                {
                    "chiplet_id": placement.chiplet_id,
                    "x": placement.x,
                    "y": placement.y,
                    "rotation": placement.rotation,
                }
                for placement in self.layout.placements
            ],
        }
