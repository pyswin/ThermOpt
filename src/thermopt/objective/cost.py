from __future__ import annotations

from dataclasses import dataclass

from thermopt.layout.objects import FloorplanCase, Layout
from thermopt.objective.metrics import collect_metrics
from thermopt.thermal.backend import ThermalBackend, build_thermal_backend


@dataclass(frozen=True)
class CostResult:
    total: float
    metrics: dict[str, float]


class Objective:
    def __init__(
        self,
        case: FloorplanCase,
        thermal_config: dict,
        objective_config: dict,
        reference_layout: Layout,
        thermal_backend: ThermalBackend | None = None,
    ):
        self.case = case
        self.thermal_config = thermal_config
        self.config = objective_config
        self.backend = thermal_backend or build_thermal_backend(case, thermal_config)
        reference = self.evaluate_raw(reference_layout)
        self.wl0 = max(reference["wirelength"], 1e-9)
        self.t0 = max(reference["thermal"], 1e-9)

    def evaluate_raw(self, layout: Layout) -> dict[str, float]:
        temperature = self.backend.simulate(self.case, layout)
        return collect_metrics(
            self.case,
            layout,
            temperature,
            thermal_mode=self.config.get("thermal_mode", "topk"),
            topk_percent=float(self.config.get("topk_percent", 0.05)),
            temperature_limit=float(self.config.get("temperature_limit", 85.0)),
        )

    def __call__(self, layout: Layout) -> CostResult:
        metrics = self.evaluate_raw(layout)
        total = (
            float(self.config.get("alpha", 1.0)) * metrics["wirelength"] / self.wl0
            + float(self.config.get("beta", 1.0)) * metrics["thermal"] / self.t0
            + float(self.config.get("gamma", 50.0)) * metrics["outline_penalty"]
            + float(self.config.get("delta", 80.0)) * metrics["overlap_penalty"]
        )
        metrics = dict(metrics)
        metrics["total_cost"] = float(total)
        return CostResult(total=float(total), metrics=metrics)
