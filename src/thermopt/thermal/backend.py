from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from thermopt.layout.objects import FloorplanCase, Layout
from thermopt.thermal.heuristic import simulate_temperature


class ThermalBackend(Protocol):
    name: str

    def simulate(self, case: FloorplanCase, layout: Layout) -> np.ndarray: ...


@dataclass(frozen=True)
class HeuristicBackend:
    config: dict
    name: str = "heuristic"
    runtime_mode: str = "heuristic"

    def simulate(self, case: FloorplanCase, layout: Layout) -> np.ndarray:
        return simulate_temperature(case, layout, self.config)


def build_thermal_backend(case: FloorplanCase, thermal_config: dict, work_dir=None) -> ThermalBackend:
    backend_name = str(thermal_config.get("backend", "heuristic")).lower()
    if backend_name == "hotspot":
        from thermopt.thermal.hotspot import HotSpotBackend

        return HotSpotBackend(case=case, config=thermal_config, work_dir=work_dir)
    return HeuristicBackend(config=thermal_config)
