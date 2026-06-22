from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from thermopt.layout.objects import FloorplanCase, Layout


class ThermalBackend(Protocol):
    name: str

    def simulate(self, case: FloorplanCase, layout: Layout) -> np.ndarray: ...


@dataclass(frozen=True)
class AIThermalBackend:
    config: dict
    name: str = "ai"
    runtime_mode: str = "ai-unimplemented"

    def simulate(self, case: FloorplanCase, layout: Layout) -> np.ndarray:
        raise NotImplementedError(
            "AI thermal backend is reserved but not implemented. Configure backend: hotspot for thermal runs."
        )


def build_thermal_backend(case: FloorplanCase, thermal_config: dict, work_dir=None) -> ThermalBackend:
    backend_name = str(thermal_config.get("backend", "hotspot")).lower()
    if backend_name == "hotspot":
        from thermopt.thermal.hotspot import HotSpotBackend

        return HotSpotBackend(case=case, config=thermal_config, work_dir=work_dir)
    if backend_name == "ai":
        return AIThermalBackend(config=thermal_config)
    raise ValueError(f"unknown thermal backend: {backend_name}. Expected 'hotspot' or 'ai'.")
