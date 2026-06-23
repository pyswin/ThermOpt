from __future__ import annotations

import numpy as np

from thermopt.layout.geometry import hpwl, total_outline_penalty, total_overlap_penalty
from thermopt.layout.objects import FloorplanCase, Layout


def topk_temperature(temperature: np.ndarray, percent: float) -> float:
    flat = np.ravel(temperature)
    k = max(1, int(round(flat.size * percent)))
    top = np.partition(flat, flat.size - k)[-k:]
    return float(np.mean(top))


def thermal_metric(temperature: np.ndarray, mode: str, topk_percent: float = 0.05, temperature_limit: float = 85.0) -> float:
    mode = mode.lower()
    if mode == "tmax":
        return float(np.max(temperature))
    if mode == "topk":
        return topk_temperature(temperature, topk_percent)
    if mode == "threshold":
        return max(0.0, float(np.max(temperature)) - temperature_limit)
    raise ValueError(f"Unknown thermal mode: {mode}")


def collect_metrics(
    case: FloorplanCase,
    layout: Layout,
    temperature: np.ndarray,
    thermal_mode: str,
    topk_percent: float,
    temperature_limit: float,
) -> dict[str, float]:
    return {
        "wirelength": hpwl(case, layout),
        "tmax": float(np.max(temperature)),
        "top1": topk_temperature(temperature, 0.01),
        "top5": topk_temperature(temperature, 0.05),
        "thermal": thermal_metric(temperature, thermal_mode, topk_percent, temperature_limit),
        "outline_penalty": total_outline_penalty(case, layout),
        "overlap_penalty": total_overlap_penalty(case, layout),
    }


def collect_layout_metrics(case: FloorplanCase, layout: Layout) -> dict[str, float]:
    return {
        "wirelength": hpwl(case, layout),
        "outline_penalty": total_outline_penalty(case, layout),
        "overlap_penalty": total_overlap_penalty(case, layout),
    }
