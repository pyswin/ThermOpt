from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from thermopt.data.atplace import load_atplace_case
from thermopt.data.case_generator import random_initial_layout
from thermopt.data.pointwise_features import build_pointwise_feature_columns
from thermopt.layout.objects import FloorplanCase, Layout, Placement
from thermopt.thermal.backend import ThermalBackend, build_thermal_backend


@dataclass(frozen=True)
class ThermalSample:
    sample_id: int
    timestamp: str
    variation_type: str
    layout_mode: str
    chiplet_configs: list[dict[str, Any]]
    temperature_map: np.ndarray
    temperature_stats: dict[str, float]


class ThermalDatasetGenerator:
    def __init__(
        self,
        case_dir: str | Path,
        thermal_config: dict | None = None,
        *,
        config_name: str = "reproduce.json",
        config_mode: str = "thermal",
        use_case_config: bool = True,
        unit_scale: float = 0.001,
        initial_layout: str = "pl",
        layout_mode: str = "random",
        min_gap: float = 0.05,
        randomize_position: bool = True,
        randomize_power: bool = True,
        randomize_rotation: bool = True,
        power_additive_fraction: float = 0.20,
        power_dropout_prob: float = 0.05,
        power_sleep_ratio: float = 0.05,
        power_shutdown_prob: float = 0.02,
        min_power_density: float = 0.0,
        max_power_density: float | None = None,
        tdp_limit: float | None = None,
        tdp_limit_ratio: float = 1.25,
        work_dir: str | Path | None = None,
        seed: int = 0,
    ) -> None:
        self.case_dir = Path(case_dir)
        self.config_name = config_name
        self.config_mode = config_mode
        self.use_case_config = bool(use_case_config)
        self.unit_scale = float(unit_scale)
        self.initial_layout = initial_layout
        self.layout_mode = self._normalize_layout_mode(layout_mode)
        self.min_gap = float(min_gap)
        self.randomize_position = bool(randomize_position)
        self.randomize_power = bool(randomize_power)
        self.randomize_rotation = bool(randomize_rotation)
        self.power_additive_fraction = float(power_additive_fraction)
        self.power_dropout_prob = float(power_dropout_prob)
        self.power_sleep_ratio = float(power_sleep_ratio)
        self.power_shutdown_prob = float(power_shutdown_prob)
        self.min_power_density = float(min_power_density)
        self.max_power_density = max_power_density
        self.tdp_limit = tdp_limit
        self.tdp_limit_ratio = float(tdp_limit_ratio)
        self.rng = np.random.default_rng(seed)

        case_input = load_atplace_case(
            self.case_dir,
            {
                "unit_scale": self.unit_scale,
                "initial_layout": self.initial_layout,
                "config_name": self.config_name,
                "config_mode": self.config_mode,
                "use_case_config": self.use_case_config,
            },
            seed=seed,
        )
        self.case_input = case_input
        self.case = case_input.case
        self.base_layout = case_input.layout
        self.base_configs = self._layout_to_configs(self.case, self.base_layout)
        self.base_total_power = float(sum(config["base_power"] for config in self.base_configs))
        self.mean_base_power = self.base_total_power / max(1, len(self.base_configs))
        self.default_tdp_limit = (
            float(self.tdp_limit)
            if self.tdp_limit is not None
            else self.base_total_power * self.tdp_limit_ratio
        )
        if self.max_power_density is None:
            densities = [
                config["base_power"] / self._area_mm2(config)
                for config in self.base_configs
                if self._area_mm2(config) > 0.0
            ]
            self.max_power_density = max(densities) * 1.5 if densities else 0.0
        else:
            self.max_power_density = float(self.max_power_density)

        self.thermal_config = dict(
            {
                "backend": "hotspot",
                "grid_size": [64, 64],
                "ambient": 25.0,
                "scale": 0.05,
                "sigma_factor": 1.0,
                "hotspot_required": True,
                "hotspot_allow_fallback": False,
            },
            **(thermal_config or {}),
        )
        backend_name = str(self.thermal_config.get("backend", "hotspot")).lower()
        if backend_name in {"thermfm", "thermfm_t", "ufno", "ufno_demo"}:
            raise ValueError(
                "Therm-FM and U-FNO are optimizer-only. Dataset generation must use HotSpot as the golden backend."
            )
        self.backend: ThermalBackend = build_thermal_backend(self.case, self.thermal_config, work_dir=work_dir)

    @staticmethod
    def _layout_to_configs(case: FloorplanCase, layout: Layout) -> list[dict[str, Any]]:
        chiplets = case.chiplet_by_id
        configs: list[dict[str, Any]] = []
        for placement in layout.placements:
            chiplet = chiplets[placement.chiplet_id]
            cx, cy = placement.center()
            configs.append(
                {
                    "name": placement.chiplet_id,
                    "cx": float(cx),
                    "cy": float(cy),
                    "x": float(cx - chiplet.width * 0.5),
                    "y": float(cy - chiplet.height * 0.5),
                    "width": float(chiplet.width),
                    "height": float(chiplet.height),
                    "base_width": float(chiplet.width),
                    "base_height": float(chiplet.height),
                    "rotation": int(placement.rotation) % 360,
                    "power": float(chiplet.power),
                    "base_power": float(chiplet.power),
                }
            )
        return configs

    def _clamp_config_position(self, config: dict[str, Any]) -> dict[str, Any]:
        max_x = max(0.0, self.case.outline_width - float(config["width"]))
        max_y = max(0.0, self.case.outline_height - float(config["height"]))
        config["x"] = float(np.clip(float(config["x"]), 0.0, max_x))
        config["y"] = float(np.clip(float(config["y"]), 0.0, max_y))
        config["cx"] = float(config["x"]) + float(config["width"]) * 0.5
        config["cy"] = float(config["y"]) + float(config["height"]) * 0.5
        return config

    @staticmethod
    def _area_mm2(config: dict[str, Any]) -> float:
        return float(config["width"]) * float(config["height"])

    @staticmethod
    def _config_center(config: dict[str, Any]) -> tuple[float, float]:
        if "cx" in config and "cy" in config:
            return float(config["cx"]), float(config["cy"])
        return (
            float(config["x"]) + float(config["width"]) * 0.5,
            float(config["y"]) + float(config["height"]) * 0.5,
        )

    @staticmethod
    def _distance(first: tuple[float, float], second: tuple[float, float]) -> float:
        return float(np.hypot(float(first[0]) - float(second[0]), float(first[1]) - float(second[1])))

    def _board_center(self) -> tuple[float, float]:
        return self.case.outline_width * 0.5, self.case.outline_height * 0.5

    def _point_boundary_distance(self, x: float, y: float) -> float:
        return float(min(float(x), float(y), self.case.outline_width - float(x), self.case.outline_height - float(y)))

    def _boundary_distance(self, config: dict[str, Any]) -> float:
        left, right, bottom, top = self._chiplet_bounds(config)
        return float(min(left, bottom, self.case.outline_width - right, self.case.outline_height - top))

    def _layout_priority(self, config: dict[str, Any], mode: str) -> tuple[float, float, float]:
        power = float(config.get("power", 0.0))
        area = self._area_mm2(config)
        if mode in {"cluster", "boundary", "dispersed"}:
            return (power, area, -float(config.get("rotation", 0)))
        return (area, power, -float(config.get("rotation", 0)))

    def _normalize_layout_mode(self, layout_mode: str) -> str:
        normalized = str(layout_mode).strip().lower()
        aliases = {
            "random": "random",
            "ordinary": "random",
            "ordinary_random": "random",
            "cluster": "cluster",
            "hot_cluster": "cluster",
            "high_temperature_cluster": "cluster",
            "boundary": "boundary",
            "boundary_pressure": "boundary",
            "edge_pressure": "boundary",
            "dispersed": "dispersed",
            "cooling": "dispersed",
            "spread": "dispersed",
            "sparse": "dispersed",
        }
        if normalized not in aliases:
            raise ValueError(
                f"unknown layout_mode: {layout_mode}. Expected one of: random, cluster, boundary, dispersed"
            )
        return aliases[normalized]

    def _candidate_from_center(self, template: dict[str, Any], center_x: float, center_y: float) -> dict[str, Any]:
        candidate = template.copy()
        half_w = float(candidate["width"]) * 0.5
        half_h = float(candidate["height"]) * 0.5
        min_cx = half_w
        max_cx = self.case.outline_width - half_w
        min_cy = half_h
        max_cy = self.case.outline_height - half_h
        center_x = float(np.clip(center_x, min_cx, max_cx))
        center_y = float(np.clip(center_y, min_cy, max_cy))
        candidate["cx"] = center_x
        candidate["cy"] = center_y
        candidate["x"] = center_x - half_w
        candidate["y"] = center_y - half_h
        return candidate

    def _ordered_templates_for_mode(self, templates: list[dict[str, Any]], mode: str) -> list[tuple[int, dict[str, Any]]]:
        indexed = list(enumerate([config.copy() for config in templates]))
        if mode == "random":
            self.rng.shuffle(indexed)
            return indexed

        indexed.sort(
            key=lambda item: (
                float(item[1].get("power", 0.0)),
                self._area_mm2(item[1]),
            ),
            reverse=True,
        )
        return indexed

    def _sample_random_target(self, template: dict[str, Any]) -> tuple[float, float]:
        mode = str(self.rng.choice(["uniform", "edge", "corner", "center"], p=[0.45, 0.25, 0.15, 0.15]))
        half_w = float(template["width"]) * 0.5
        half_h = float(template["height"]) * 0.5
        cx_low = half_w
        cx_high = self.case.outline_width - half_w
        cy_low = half_h
        cy_high = self.case.outline_height - half_h
        board_cx, board_cy = self._board_center()
        edge_band = max(self.min_gap * 2.0, min(self.case.outline_width, self.case.outline_height) * 0.08)

        if mode == "uniform":
            return (
                float(self.rng.uniform(cx_low, cx_high)),
                float(self.rng.uniform(cy_low, cy_high)),
            )
        if mode == "edge":
            edge = str(self.rng.choice(["left", "right", "bottom", "top"]))
            if edge == "left":
                return (cx_low + min(edge_band, cx_high - cx_low) * 0.25, float(self.rng.uniform(cy_low, cy_high)))
            if edge == "right":
                return (cx_high - min(edge_band, cx_high - cx_low) * 0.25, float(self.rng.uniform(cy_low, cy_high)))
            if edge == "bottom":
                return (float(self.rng.uniform(cx_low, cx_high)), cy_low + min(edge_band, cy_high - cy_low) * 0.25)
            return (float(self.rng.uniform(cx_low, cx_high)), cy_high - min(edge_band, cy_high - cy_low) * 0.25)
        if mode == "corner":
            corner = str(self.rng.choice(["ll", "lr", "ul", "ur"]))
            x = cx_low if corner in {"ll", "ul"} else cx_high
            y = cy_low if corner in {"ll", "lr"} else cy_high
            return (x, y)
        return (
            float(np.clip(self.rng.normal(board_cx, self.case.outline_width * 0.12), cx_low, cx_high)),
            float(np.clip(self.rng.normal(board_cy, self.case.outline_height * 0.12), cy_low, cy_high)),
        )

    def _sample_mode_targets(
        self,
        mode: str,
        ordered_templates: list[tuple[int, dict[str, Any]]],
    ) -> list[tuple[float, float]]:
        total = len(ordered_templates)
        if total == 0:
            return []

        mode = self._normalize_layout_mode(mode)
        board_cx, board_cy = self._board_center()
        targets: list[tuple[float, float]] = []

        if mode == "random":
            return [self._sample_random_target(template) for _, template in ordered_templates]

        if mode == "cluster":
            anchor_x = float(np.clip(self.rng.normal(board_cx, self.case.outline_width * 0.08), 0.0, self.case.outline_width))
            anchor_y = float(np.clip(self.rng.normal(board_cy, self.case.outline_height * 0.08), 0.0, self.case.outline_height))
            radius_base = max(self.min_gap * 4.0, min(self.case.outline_width, self.case.outline_height) * 0.16)
            angle_offset = float(self.rng.uniform(0.0, 2.0 * np.pi))
            for rank, (_, template) in enumerate(ordered_templates):
                fraction = rank / max(total - 1, 1)
                radius = radius_base * (0.10 + 0.75 * fraction)
                theta = angle_offset + 2.0 * np.pi * (0.61803398875 * rank)
                cx = anchor_x + radius * float(np.cos(theta))
                cy = anchor_y + radius * float(np.sin(theta))
                targets.append(self._candidate_from_center(template, cx, cy)["cx"],)
            return [
                (
                    float(np.clip(anchor_x + radius_base * (0.10 + 0.75 * (rank / max(total - 1, 1))) * float(np.cos(angle_offset + 2.0 * np.pi * (0.61803398875 * rank))), 0.0, self.case.outline_width)),
                    float(np.clip(anchor_y + radius_base * (0.10 + 0.75 * (rank / max(total - 1, 1))) * float(np.sin(angle_offset + 2.0 * np.pi * (0.61803398875 * rank))), 0.0, self.case.outline_height)),
                )
                for rank, (_, template) in enumerate(ordered_templates)
            ]

        if mode == "boundary":
            edge = str(self.rng.choice(["left", "right", "bottom", "top"]))
            band = max(self.min_gap * 2.0, min(self.case.outline_width, self.case.outline_height) * 0.08)
            span = np.linspace(0.15, 0.85, total)
            self.rng.shuffle(span)
            for rank, (_, template) in enumerate(ordered_templates):
                if edge in {"left", "right"}:
                    y = float(np.clip(span[rank] * self.case.outline_height + self.rng.normal(0.0, self.case.outline_height * 0.025), 0.0, self.case.outline_height))
                    if rank < max(1, total // 3):
                        x = band if edge == "left" else self.case.outline_width - band
                    else:
                        inner = band * (0.55 + 0.35 * self.rng.random())
                        x = inner if edge == "left" else self.case.outline_width - inner
                else:
                    x = float(np.clip(span[rank] * self.case.outline_width + self.rng.normal(0.0, self.case.outline_width * 0.025), 0.0, self.case.outline_width))
                    if rank < max(1, total // 3):
                        y = band if edge == "bottom" else self.case.outline_height - band
                    else:
                        inner = band * (0.55 + 0.35 * self.rng.random())
                        y = inner if edge == "bottom" else self.case.outline_height - inner
                targets.append(self._candidate_from_center(template, x, y)["cx"],)
            return [
                (
                    float(np.clip(self._candidate_from_center(template, 0.0, 0.0)["cx"], 0.0, self.case.outline_width)),
                    float(np.clip(self._candidate_from_center(template, 0.0, 0.0)["cy"], 0.0, self.case.outline_height)),
                )
                for _, template in ordered_templates
            ] if False else [
                (
                    float(np.clip(span[rank] * self.case.outline_width if edge in {"bottom", "top"} else (band if edge == "left" else self.case.outline_width - band), 0.0, self.case.outline_width)),
                    float(np.clip(span[rank] * self.case.outline_height if edge in {"left", "right"} else (band if edge == "bottom" else self.case.outline_height - band), 0.0, self.case.outline_height)),
                )
                for rank, (_, template) in enumerate(ordered_templates)
            ]

        anchor_count = min(total, max(4, int(np.ceil(np.sqrt(total)))))
        anchors = self._farthest_point_anchors(
            anchor_count,
            0.0,
            self.case.outline_width,
            0.0,
            self.case.outline_height,
            candidate_pool=256,
        )
        if not anchors:
            anchors = [self._board_center()]
        jitter = max(self.min_gap * 2.0, min(self.case.outline_width, self.case.outline_height) * 0.08)
        for rank, (_, template) in enumerate(ordered_templates):
            anchor = anchors[rank % len(anchors)]
            cx = float(np.clip(self.rng.normal(anchor[0], jitter), 0.0, self.case.outline_width))
            cy = float(np.clip(self.rng.normal(anchor[1], jitter), 0.0, self.case.outline_height))
            targets.append(self._candidate_from_center(template, cx, cy)["cx"],)
        return [
            (
                float(np.clip(self.rng.normal(anchors[rank % len(anchors)][0], jitter), 0.0, self.case.outline_width)),
                float(np.clip(self.rng.normal(anchors[rank % len(anchors)][1], jitter), 0.0, self.case.outline_height)),
            )
            for rank in range(total)
        ]

    def _min_center_distance(self, candidate: dict[str, Any], placed: list[dict[str, Any]]) -> float:
        if not placed:
            return min(self.case.outline_width, self.case.outline_height)
        center = self._config_center(candidate)
        return min(self._distance(center, self._config_center(other)) for other in placed)

    def _search_mode_candidate(
        self,
        template: dict[str, Any],
        target: tuple[float, float],
        mode: str,
        placed: list[dict[str, Any]],
        rank: int,
        total: int,
        candidate_count: int,
    ) -> dict[str, Any] | None:
        base_sigma = {
            "random": max(self.min_gap * 4.0, min(self.case.outline_width, self.case.outline_height) * 0.18),
            "cluster": max(self.min_gap * 3.0, min(self.case.outline_width, self.case.outline_height) * 0.10),
            "boundary": max(self.min_gap * 3.0, min(self.case.outline_width, self.case.outline_height) * 0.09),
            "dispersed": max(self.min_gap * 3.0, min(self.case.outline_width, self.case.outline_height) * 0.12),
        }[mode]
        best: dict[str, Any] | None = None
        best_score = -float("inf")

        for attempt in range(candidate_count):
            scale = 1.0 + 0.035 * attempt
            if mode == "random" and float(self.rng.random()) < 0.25:
                center_x, center_y = self._sample_random_target(template)
            else:
                center_x = float(self.rng.normal(target[0], base_sigma * scale))
                center_y = float(self.rng.normal(target[1], base_sigma * scale))
            candidate = self._candidate_from_center(template, center_x, center_y)
            if not self._is_legal_against(candidate, placed, self.min_gap):
                continue

            center = self._config_center(candidate)
            target_distance = self._distance(center, target)
            edge_distance = self._boundary_distance(candidate)
            pair_distance = self._min_center_distance(candidate, placed)
            clearance = self._clearance_to_layout(candidate, placed)
            score = 0.0

            if mode == "cluster":
                score = -1.15 * target_distance + 0.04 * clearance + 0.03 * pair_distance
            elif mode == "boundary":
                score = -1.10 * target_distance - 0.65 * edge_distance + 0.05 * clearance
            elif mode == "dispersed":
                score = -0.55 * target_distance + 0.55 * pair_distance + 0.03 * clearance
            else:
                score = -0.22 * target_distance + 0.18 * pair_distance + 0.05 * clearance

            score += float(self.rng.uniform(0.0, 0.35))
            if score > best_score:
                best = candidate
                best_score = score

        return best

    def _farthest_point_anchors(
        self,
        count: int,
        x_low: float,
        x_high: float,
        y_low: float,
        y_high: float,
        *,
        candidate_pool: int = 128,
    ) -> list[tuple[float, float]]:
        if count <= 0:
            return []
        anchors: list[tuple[float, float]] = []
        pool_size = max(32, int(candidate_pool))
        for _ in range(count):
            pool = self.rng.uniform(
                low=np.array([x_low, y_low], dtype=float),
                high=np.array([x_high, y_high], dtype=float),
                size=(pool_size, 2),
            )
            if not anchors:
                pick = pool[int(self.rng.integers(0, len(pool)))]
            else:
                distances = []
                for candidate in pool:
                    min_distance = min(
                        self._distance((float(candidate[0]), float(candidate[1])), anchor)
                        for anchor in anchors
                    )
                    distances.append(min_distance)
                pick = pool[int(np.argmax(distances))]
            anchors.append((float(pick[0]), float(pick[1])))
        return anchors

    def _extreme_point_anchors(
        self,
        count: int,
        x_low: float,
        x_high: float,
        y_low: float,
        y_high: float,
        *,
        candidate_pool: int = 128,
    ) -> list[tuple[float, float]]:
        if count <= 0:
            return []
        anchors = [
            (x_low, y_low),
            (x_low, y_high),
            (x_high, y_low),
            (x_high, y_high),
            ((x_low + x_high) * 0.5, y_low),
            ((x_low + x_high) * 0.5, y_high),
            (x_low, (y_low + y_high) * 0.5),
            (x_high, (y_low + y_high) * 0.5),
        ]
        if count <= len(anchors):
            return anchors[:count]

        # Add farthest points inside the remaining space.
        while len(anchors) < count:
            pool = self.rng.uniform(
                low=np.array([x_low, y_low], dtype=float),
                high=np.array([x_high, y_high], dtype=float),
                size=(max(32, int(candidate_pool)), 2),
            )
            pick = None
            best_score = -float("inf")
            for candidate in pool:
                score = min(self._distance((float(candidate[0]), float(candidate[1])), anchor) for anchor in anchors)
                score += float(self.rng.uniform(0.0, 0.05))
                if score > best_score:
                    pick = candidate
                    best_score = score
            anchors.append((float(pick[0]), float(pick[1])))
        return anchors

    def _boundary_point_anchors(
        self,
        count: int,
        x_low: float,
        x_high: float,
        y_low: float,
        y_high: float,
        *,
        candidate_pool: int = 128,
    ) -> list[tuple[float, float]]:
        if count <= 0:
            return []
        anchors: list[tuple[float, float]] = []
        pool_size = max(32, int(candidate_pool))
        for _ in range(count):
            pool = np.empty((pool_size, 2), dtype=np.float64)
            for idx in range(pool_size):
                edge = str(self.rng.choice(["left", "right", "bottom", "top"]))
                if edge == "left":
                    pool[idx] = (x_low, float(self.rng.uniform(y_low, y_high)))
                elif edge == "right":
                    pool[idx] = (x_high, float(self.rng.uniform(y_low, y_high)))
                elif edge == "bottom":
                    pool[idx] = (float(self.rng.uniform(x_low, x_high)), y_low)
                else:
                    pool[idx] = (float(self.rng.uniform(x_low, x_high)), y_high)
            if not anchors:
                pick = pool[int(self.rng.integers(0, len(pool)))]
            else:
                scores = []
                for candidate in pool:
                    distances = [self._distance((float(candidate[0]), float(candidate[1])), anchor) for anchor in anchors]
                    scores.append(min(distances))
                pick = pool[int(np.argmax(scores))]
            anchors.append((float(pick[0]), float(pick[1])))
        return anchors

    @staticmethod
    def _balanced_group_sizes(total: int, group_count: int) -> list[int]:
        if total <= 0:
            return []
        group_count = max(1, min(int(group_count), total))
        return [total // group_count + (1 if index < (total % group_count) else 0) for index in range(group_count)]

    def _shuffle_same_priority_groups(
        self,
        indexed: list[tuple[int, dict[str, Any]]],
        *,
        primary: str,
    ) -> list[tuple[int, dict[str, Any]]]:
        grouped: dict[tuple[float, float], list[tuple[int, dict[str, Any]]]] = {}
        for item in indexed:
            template = item[1]
            power = round(float(template.get("power", 0.0)), 6)
            area = round(self._area_mm2(template), 6)
            key = (power, area) if primary == "power_area" else (area, power)
            grouped.setdefault(key, []).append(item)

        ordered: list[tuple[int, dict[str, Any]]] = []
        for key in sorted(grouped.keys(), reverse=True):
            group = grouped[key]
            if len(group) > 1:
                self.rng.shuffle(group)
            ordered.extend(group)
        return ordered

    def _ordered_templates_for_slot_layout(
        self,
        templates: list[dict[str, Any]],
        mode: str,
    ) -> list[tuple[int, dict[str, Any]]]:
        indexed = list(enumerate([config.copy() for config in templates]))
        mode = self._normalize_layout_mode(mode)
        if mode in {"cluster", "boundary"}:
            indexed.sort(
                key=lambda item: (
                    float(item[1].get("power", 0.0)),
                    self._area_mm2(item[1]),
                ),
                reverse=True,
            )
            return self._shuffle_same_priority_groups(indexed, primary="power_area")
        if mode == "dispersed":
            indexed.sort(
                key=lambda item: (
                    self._area_mm2(item[1]),
                    float(item[1].get("power", 0.0)),
                ),
                reverse=True,
            )
            if len(indexed) > 3:
                bucket_count = min(len(indexed), max(3, int(np.ceil(len(indexed) / 4))))
                bucket_sizes = self._balanced_group_sizes(len(indexed), bucket_count)
                buckets: list[list[tuple[int, dict[str, Any]]]] = []
                cursor = 0
                for size in bucket_sizes:
                    bucket = indexed[cursor : cursor + size]
                    cursor += size
                    if len(bucket) > 1:
                        self.rng.shuffle(bucket)
                    buckets.append(bucket)
                self.rng.shuffle(buckets)
                spread: list[tuple[int, dict[str, Any]]] = []
                for bucket in buckets:
                    spread.extend(bucket)
                return spread
            return indexed
        indexed.sort(
            key=lambda item: (
                self._area_mm2(item[1]),
                float(item[1].get("power", 0.0)),
            ),
            reverse=True,
        )
        if mode == "random" and len(indexed) > 3 and float(self.rng.random()) < 0.45:
            front = indexed[::2]
            back = indexed[1::2]
            indexed = front + back[::-1]
        elif mode == "dispersed" and len(indexed) > 3:
            ordered: list[tuple[int, dict[str, Any]]] = []
            left = 0
            right = len(indexed) - 1
            take_left = True
            while left <= right:
                if take_left:
                    ordered.append(indexed[left])
                    left += 1
                else:
                    ordered.append(indexed[right])
                    right -= 1
                take_left = not take_left
            indexed = ordered
        return indexed

    def _random_targets_from_initial_layout(
        self,
        ordered_templates: list[tuple[int, dict[str, Any]]],
    ) -> list[tuple[float, float]]:
        seed = int(self.rng.integers(0, 2**32 - 1))
        random_layout = random_initial_layout(self.case, seed)
        placements = list(random_layout.placements)
        if not placements:
            return []
        self.rng.shuffle(placements)

        targets: list[tuple[float, float]] = []
        for (original_index, template), placement in zip(ordered_templates, placements):
            chiplet = self.case.chiplet_by_id[placement.chiplet_id]
            candidate = template.copy()
            center_x = float(placement.x)
            center_y = float(placement.y)
            candidate["width"] = float(template["width"])
            candidate["height"] = float(template["height"])
            candidate["x"] = center_x - float(candidate["width"]) * 0.5
            candidate["y"] = center_y - float(candidate["height"]) * 0.5
            candidate = self._clamp_config_position(candidate)
            targets.append((float(candidate["cx"]), float(candidate["cy"])))
        return targets

    def _staggered_row_targets(
        self,
        ordered_templates: list[tuple[int, dict[str, Any]]],
    ) -> list[tuple[float, float]]:
        total = len(ordered_templates)
        if total == 0:
            return []

        max_width = max(float(template["width"]) for _, template in ordered_templates)
        max_height = max(float(template["height"]) for _, template in ordered_templates)
        row_count = min(total, max(2, int(round(np.sqrt(total)))))
        if total >= 10 and row_count < 4 and float(self.rng.random()) < 0.45:
            row_count += 1
        row_count = min(row_count, total)

        row_sizes = self._balanced_group_sizes(total, row_count)
        row_indices = list(range(row_count))
        row_indices.sort(key=lambda idx: abs(idx - (row_count - 1) * 0.5))

        x_margin = max(self.min_gap, max_width * 0.5 + self.min_gap)
        y_margin = max(self.min_gap, max_height * 0.5 + self.min_gap)
        x_low = x_margin
        x_high = self.case.outline_width - x_margin
        y_low = y_margin
        y_high = self.case.outline_height - y_margin
        if x_high <= x_low:
            x_low, x_high = max(0.0, self.min_gap), self.case.outline_width - max(0.0, self.min_gap)
        if y_high <= y_low:
            y_low, y_high = max(0.0, self.min_gap), self.case.outline_height - max(0.0, self.min_gap)

        row_centers = np.linspace(y_low, y_high, row_count, dtype=np.float64)
        row_centers += self.rng.normal(0.0, max(self.min_gap * 1.5, (y_high - y_low) * 0.06), size=row_count)
        row_centers = np.clip(row_centers, y_low, y_high)
        row_centers.sort()

        slot_centers: list[tuple[float, float]] = []
        for row_idx in row_indices:
            count = row_sizes[row_idx]
            if count <= 0:
                continue
            if count == 1:
                xs = np.array([(x_low + x_high) * 0.5], dtype=np.float64)
            else:
                xs = np.linspace(x_low, x_high, count, dtype=np.float64)
                step = (x_high - x_low) / max(count - 1, 1)
                phase = (row_idx - (row_count - 1) * 0.5) * step * 0.18
                phase += float(self.rng.normal(0.0, max(self.min_gap, step * 0.08)))
                if row_idx % 2 == 1:
                    phase += step * 0.22
                xs = xs + phase
                xs += self.rng.normal(0.0, max(self.min_gap * 0.6, step * 0.07), size=count)
                xs = np.clip(xs, x_low, x_high)
                xs.sort()
                if row_idx % 3 == 2:
                    xs = xs[::-1]
            slot_centers.extend((float(x), float(row_centers[row_idx])) for x in xs)

        targets: list[tuple[float, float]] = []
        for (original_index, template), (slot_x, slot_y) in zip(ordered_templates, slot_centers):
            candidate = self._candidate_from_center(template, slot_x, slot_y)
            targets.append((float(candidate["cx"]), float(candidate["cy"])))
        return targets

    def _anchor_island_targets(
        self,
        ordered_templates: list[tuple[int, dict[str, Any]]],
        anchor_count: int,
        *,
        spread_scale: float,
        cluster_scale: float,
        anchor_priority: str = "center",
    ) -> list[tuple[float, float]]:
        total = len(ordered_templates)
        if total == 0:
            return []

        max_width = max(float(template["width"]) for _, template in ordered_templates)
        max_height = max(float(template["height"]) for _, template in ordered_templates)
        anchor_count = min(total, max(1, int(anchor_count)))
        anchor_count = max(1, min(anchor_count, total))
        group_sizes = self._balanced_group_sizes(total, anchor_count)

        x_margin = max(self.min_gap, max_width * 0.5 + self.min_gap)
        y_margin = max(self.min_gap, max_height * 0.5 + self.min_gap)
        x_low = x_margin
        x_high = self.case.outline_width - x_margin
        y_low = y_margin
        y_high = self.case.outline_height - y_margin
        if x_high <= x_low:
            x_low, x_high = max(0.0, self.min_gap), self.case.outline_width - max(0.0, self.min_gap)
        if y_high <= y_low:
            y_low, y_high = max(0.0, self.min_gap), self.case.outline_height - max(0.0, self.min_gap)

        if anchor_priority == "boundary":
            anchors = self._boundary_point_anchors(anchor_count, x_low, x_high, y_low, y_high, candidate_pool=256)
        elif anchor_priority == "extreme":
            anchors = self._extreme_point_anchors(anchor_count, x_low, x_high, y_low, y_high, candidate_pool=256)
        else:
            anchors = self._farthest_point_anchors(anchor_count, x_low, x_high, y_low, y_high, candidate_pool=256)
        if not anchors:
            anchors = [self._board_center()]
        anchor_order = list(range(len(anchors)))
        if anchor_priority == "boundary":
            anchor_order.sort(key=lambda idx: self._point_boundary_distance(anchors[idx][0], anchors[idx][1]))
        elif anchor_priority == "center":
            anchor_order.sort(key=lambda idx: self._distance(anchors[idx], self._board_center()))
        else:
            self.rng.shuffle(anchor_order)

        local_step = max(max_width, max_height) + self.min_gap * 2.0 + min(self.case.outline_width, self.case.outline_height) * spread_scale
        ring_step = max(self.min_gap * 1.5, local_step * cluster_scale)
        slot_centers: list[tuple[float, float]] = []
        for group_index, anchor_idx in enumerate(anchor_order):
            anchor_x, anchor_y = anchors[anchor_idx]
            count = group_sizes[group_index]
            if count <= 0:
                continue
            base_angle = float(self.rng.uniform(0.0, 2.0 * np.pi))
            for slot_index in range(count):
                if slot_index == 0:
                    offset_x = 0.0
                    offset_y = 0.0
                else:
                    radius = ring_step * (0.55 + 0.30 * (slot_index - 1))
                    angle = base_angle + 2.0 * np.pi * (0.61803398875 * slot_index)
                    offset_x = radius * float(np.cos(angle))
                    offset_y = radius * float(np.sin(angle))
                jitter = max(self.min_gap * 0.5, ring_step * 0.10)
                slot_centers.append(
                    (
                        float(np.clip(anchor_x + offset_x + self.rng.normal(0.0, jitter), x_low, x_high)),
                        float(np.clip(anchor_y + offset_y + self.rng.normal(0.0, jitter), y_low, y_high)),
                    )
                )

        targets: list[tuple[float, float]] = []
        for (original_index, template), (slot_x, slot_y) in zip(ordered_templates, slot_centers):
            candidate = self._candidate_from_center(template, slot_x, slot_y)
            targets.append((float(candidate["cx"]), float(candidate["cy"])))
        return targets

    def _build_representative_targets(
        self,
        mode: str,
        ordered_templates: list[tuple[int, dict[str, Any]]],
    ) -> list[tuple[float, float]]:
        mode = self._normalize_layout_mode(mode)
        if mode == "random":
            if float(self.rng.random()) < 0.65:
                return self._random_targets_from_initial_layout(ordered_templates)
            return self._staggered_row_targets(ordered_templates)
        if mode == "dispersed":
            anchor_count = min(len(ordered_templates), max(4, int(np.ceil(len(ordered_templates) / 3))))
            return self._anchor_island_targets(
                ordered_templates,
                anchor_count,
                spread_scale=0.11,
                cluster_scale=0.88,
                anchor_priority="extreme",
            )
        if mode == "cluster":
            return self._anchor_island_targets(
                ordered_templates,
                anchor_count=min(len(ordered_templates), max(1, int(np.ceil(len(ordered_templates) / 5)))),
                spread_scale=0.015,
                cluster_scale=0.25,
                anchor_priority="center",
            )
        if mode == "boundary":
            return self._anchor_island_targets(
                ordered_templates,
                anchor_count=min(len(ordered_templates), max(2, int(np.ceil(len(ordered_templates) / 5)))),
                spread_scale=0.03,
                cluster_scale=0.34,
                anchor_priority="boundary",
            )
        raise ValueError(f"unknown layout_mode: {mode}")

    def _generate_representative_layout(
        self,
        mode: str,
        templates: list[dict[str, Any]] | None = None,
        layout_attempts: int = 80,
        candidate_count: int = 256,
    ) -> list[dict[str, Any]]:
        mode = self._normalize_layout_mode(mode)
        if templates is None:
            templates = self.base_configs
        templates = [config.copy() for config in templates]
        best_partial: list[dict[str, Any]] = []

        for _ in range(layout_attempts):
            ordered_templates = self._ordered_templates_for_slot_layout(templates, mode)
            targets = self._build_representative_targets(mode, ordered_templates)
            free_rects: list[tuple[float, float, float, float]] = [
                (0.0, 0.0, float(self.case.outline_width), float(self.case.outline_height))
            ]
            placed: list[dict[str, Any]] = []
            by_index: dict[int, dict[str, Any]] = {}

            for (original_index, template), target in zip(ordered_templates, targets):
                chosen = self._select_free_rect_for_target(free_rects, template, target, mode, placed)
                if chosen is None:
                    break
                rect_index, candidate, padded_rect = chosen
                free_rect = free_rects.pop(rect_index)
                free_rects.extend(self._split_free_rect(free_rect, padded_rect))
                free_rects = self._prune_free_rects(free_rects)
                placed.append(candidate)
                by_index[original_index] = candidate

            if len(placed) > len(best_partial):
                best_partial = placed
            if len(by_index) == len(templates):
                return [by_index[idx] for idx in range(len(templates))]

        raise RuntimeError(
            f"无法生成满足最小间距 {self.min_gap}mm 的 chiplet 布局；"
            f"最多只成功放置 {len(best_partial)}/{len(templates)} 个 chiplet。"
        )

    def _layout_mode_context(self, layout_mode: str, templates: list[dict[str, Any]]) -> dict[str, Any]:
        mode = self._normalize_layout_mode(layout_mode)
        max_width = max(float(config["width"]) for config in templates)
        max_height = max(float(config["height"]) for config in templates)
        x_low = max(self.min_gap, max_width * 0.5 + self.min_gap)
        y_low = max(self.min_gap, max_height * 0.5 + self.min_gap)
        x_high = max(x_low, self.case.outline_width - x_low)
        y_high = max(y_low, self.case.outline_height - y_low)

        if mode == "cluster":
            board_cx, board_cy = self._board_center()
            center_jitter_x = max(self.min_gap * 2.0, self.case.outline_width * 0.12)
            center_jitter_y = max(self.min_gap * 2.0, self.case.outline_height * 0.12)
            anchor_x = float(np.clip(self.rng.normal(board_cx, center_jitter_x * 0.35), x_low, x_high))
            anchor_y = float(np.clip(self.rng.normal(board_cy, center_jitter_y * 0.35), y_low, y_high))
            return {
                "anchor": (anchor_x, anchor_y),
                "cluster_radius": max(self.min_gap * 4.0, min(self.case.outline_width, self.case.outline_height) * 0.20),
            }

        if mode == "boundary":
            edge = str(self.rng.choice(["left", "right", "bottom", "top"]))
            corner = float(self.rng.random()) < 0.35
            if edge in {"left", "right"}:
                anchor_y = float(self.rng.uniform(y_low, y_high))
                anchor_x = x_low if edge == "left" else x_high
                if corner:
                    anchor_y = y_low if self.rng.random() < 0.5 else y_high
            else:
                anchor_x = float(self.rng.uniform(x_low, x_high))
                anchor_y = y_low if edge == "bottom" else y_high
                if corner:
                    anchor_x = x_low if self.rng.random() < 0.5 else x_high
            return {
                "edge": edge,
                "anchor": (float(anchor_x), float(anchor_y)),
                "corner": corner,
                "boundary_band": max(self.min_gap * 2.0, min(self.case.outline_width, self.case.outline_height) * 0.08),
            }

        if mode == "dispersed":
            return {
                "anchors": self._farthest_point_anchors(len(templates), x_low, x_high, y_low, y_high),
                "dispersion_radius": max(self.min_gap * 4.0, min(self.case.outline_width, self.case.outline_height) * 0.16),
            }

        return {}

    def _build_candidate_from_anchor(
        self,
        template: dict[str, Any],
        anchor: tuple[float, float],
        *,
        center_jitter: float,
        x_low: float = 0.0,
        x_high: float | None = None,
        y_low: float = 0.0,
        y_high: float | None = None,
    ) -> dict[str, Any]:
        x_high = self.case.outline_width if x_high is None else x_high
        y_high = self.case.outline_height if y_high is None else y_high
        candidate = template.copy()
        candidate["x"] = float(self.rng.normal(anchor[0], center_jitter)) - float(candidate["width"]) * 0.5
        candidate["y"] = float(self.rng.normal(anchor[1], center_jitter)) - float(candidate["height"]) * 0.5
        candidate = self._clamp_config_position(candidate)
        candidate["x"] = float(np.clip(candidate["x"], x_low, max(x_low, x_high - float(candidate["width"]))))
        candidate["y"] = float(np.clip(candidate["y"], y_low, max(y_low, y_high - float(candidate["height"]))))
        candidate["cx"] = candidate["x"] + float(candidate["width"]) * 0.5
        candidate["cy"] = candidate["y"] + float(candidate["height"]) * 0.5
        return candidate

    def _build_candidate_random(self, template: dict[str, Any]) -> dict[str, Any]:
        candidate = template.copy()
        max_x = max(0.0, self.case.outline_width - float(candidate["width"]))
        max_y = max(0.0, self.case.outline_height - float(candidate["height"]))
        candidate["x"] = float(self.rng.uniform(0.0, max_x))
        candidate["y"] = float(self.rng.uniform(0.0, max_y))
        return self._clamp_config_position(candidate)

    def _build_candidate_boundary(
        self,
        template: dict[str, Any],
        edge: str,
        band: float,
        corner: bool = False,
    ) -> dict[str, Any]:
        candidate = template.copy()
        max_x = max(0.0, self.case.outline_width - float(candidate["width"]))
        max_y = max(0.0, self.case.outline_height - float(candidate["height"]))
        x_anchor_low = 0.0
        x_anchor_high = max_x
        y_anchor_low = 0.0
        y_anchor_high = max_y
        if edge == "left":
            candidate["x"] = float(self.rng.uniform(x_anchor_low, min(max_x, band)))
            candidate["y"] = float(self.rng.uniform(y_anchor_low, y_anchor_high))
        elif edge == "right":
            candidate["x"] = float(self.rng.uniform(max(x_anchor_low, max_x - band), x_anchor_high))
            candidate["y"] = float(self.rng.uniform(y_anchor_low, y_anchor_high))
        elif edge == "bottom":
            candidate["x"] = float(self.rng.uniform(x_anchor_low, x_anchor_high))
            candidate["y"] = float(self.rng.uniform(y_anchor_low, min(max_y, band)))
        else:
            candidate["x"] = float(self.rng.uniform(x_anchor_low, x_anchor_high))
            candidate["y"] = float(self.rng.uniform(max(y_anchor_low, max_y - band), y_anchor_high))

        if corner:
            if edge in {"left", "right"}:
                candidate["y"] = float(self.rng.choice([y_anchor_low, y_anchor_high]))
            else:
                candidate["x"] = float(self.rng.choice([x_anchor_low, x_anchor_high]))
        return self._clamp_config_position(candidate)

    def _build_candidate_for_mode(
        self,
        template: dict[str, Any],
        mode: str,
        context: dict[str, Any],
        rank: int,
    ) -> dict[str, Any]:
        if mode == "cluster":
            anchor = context["anchor"]
            radius = float(context["cluster_radius"])
            center_jitter = max(self.min_gap * 0.5, radius * (0.18 + 0.04 * min(rank, 4)))
            return self._build_candidate_from_anchor(template, anchor, center_jitter=center_jitter)
        if mode == "boundary":
            anchor = context["anchor"]
            band = float(context["boundary_band"])
            edge = str(context["edge"])
            corner = bool(context.get("corner", False))
            candidate = self._build_candidate_boundary(template, edge=edge, band=band, corner=corner)
            edge_center = self._config_center(candidate)
            if self._distance(edge_center, anchor) > band * 3.0:
                candidate = self._build_candidate_from_anchor(
                    template,
                    anchor,
                    center_jitter=max(self.min_gap * 0.5, band * 0.6),
                )
            return candidate
        if mode == "dispersed":
            anchors = context["anchors"]
            anchor = anchors[min(rank, len(anchors) - 1)] if anchors else self._board_center()
            radius = float(context["dispersion_radius"])
            center_jitter = max(self.min_gap * 0.5, radius * 0.22)
            return self._build_candidate_from_anchor(template, anchor, center_jitter=center_jitter)
        return self._build_candidate_random(template)

    def _ordered_templates(self, templates: list[dict[str, Any]], mode: str) -> list[tuple[int, dict[str, Any]]]:
        indexed = list(enumerate(templates))
        reverse = True
        if mode == "dispersed":
            indexed.sort(key=lambda item: self._layout_priority(item[1], mode), reverse=reverse)
            if len(indexed) > 2:
                front = indexed[::2]
                back = indexed[1::2]
                ordered = front + back[::-1]
                return ordered
            return indexed
        if mode == "boundary":
            indexed.sort(key=lambda item: self._layout_priority(item[1], mode), reverse=reverse)
            return indexed
        if mode == "cluster":
            indexed.sort(key=lambda item: self._layout_priority(item[1], mode), reverse=reverse)
            return indexed
        self.rng.shuffle(indexed)
        return indexed

    def _generate_mode_layout(
        self,
        mode: str,
        templates: list[dict[str, Any]] | None = None,
        layout_attempts: int = 80,
        candidate_count: int = 256,
    ) -> list[dict[str, Any]]:
        mode = self._normalize_layout_mode(mode)
        if templates is None:
            templates = self.base_configs
        if mode in {"cluster", "boundary"}:
            topology_errors: list[str] = []
            try:
                return self._generate_topology_layout(
                    mode,
                    templates=templates,
                    layout_attempts=layout_attempts,
                    candidate_count=candidate_count,
                )
            except RuntimeError as exc:
                topology_errors.append(str(exc))

            try:
                return self._generate_packed_topology_layout(
                    mode,
                    templates=templates,
                    layout_attempts=layout_attempts,
                )
            except RuntimeError as exc:
                topology_errors.append(str(exc))

            try:
                return self._generate_representative_layout(
                    mode,
                    templates=templates,
                    layout_attempts=layout_attempts,
                    candidate_count=candidate_count,
                )
            except RuntimeError as exc:
                topology_errors.append(str(exc))
            try:
                fallback_pos_range = max(
                    self.min_gap * 4.0,
                    min(self.case.outline_width, self.case.outline_height) * (0.12 if mode == "cluster" else 0.10),
                )
                return self._generate_legal_layout(
                    fallback_pos_range,
                    self.min_gap,
                    templates=templates,
                    layout_attempts=max(layout_attempts, len(templates) * 4),
                )
            except RuntimeError as exc:
                topology_errors.append(str(exc))
                raise RuntimeError(topology_errors[-1]) from exc
        return self._generate_representative_layout(
            mode,
            templates=templates,
            layout_attempts=layout_attempts,
            candidate_count=candidate_count,
        )

    def _generate_layout_variation(
        self,
        layout_mode: str,
        pos_range: float = 2.0,
        power_range: tuple[float, float] = (0.5, 1.5),
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_layout_mode(layout_mode)
        configs = [config.copy() for config in self.base_configs]

        if self.randomize_rotation:
            configs = self._apply_random_rotation(configs)
        else:
            configs = [self._rotate_config(config, config.get("rotation", 0)) for config in configs]

        if self.randomize_position:
            try:
                configs = self._generate_mode_layout(normalized_mode, templates=configs)
            except RuntimeError:
                configs = self._generate_fallback_layout(configs, self.min_gap)
        else:
            configs = [self._clamp_config_position(config) for config in configs]

        if self.randomize_power and normalized_mode not in {"cluster", "boundary"}:
            configs = self._apply_power_variation(configs, power_range)
        else:
            for config in configs:
                config["power"] = float(config.get("base_power", config["power"]))

        valid, message = self._check_chiplet_collision(configs, self.min_gap)
        if not valid:
            raise RuntimeError(f"随机布局生成失败: {message}")
        return configs

    def _packing_order_for_mode(
        self,
        templates: list[dict[str, Any]],
        mode: str,
    ) -> list[tuple[int, dict[str, Any]]]:
        indexed = list(enumerate([config.copy() for config in templates]))
        mode = self._normalize_layout_mode(mode)
        if mode == "random":
            self.rng.shuffle(indexed)
            return indexed
        if mode in {"cluster", "boundary"}:
            indexed.sort(
                key=lambda item: (
                    float(item[1].get("power", 0.0)),
                    self._area_mm2(item[1]),
                ),
                reverse=True,
            )
            return indexed
        indexed.sort(
            key=lambda item: (
                self._area_mm2(item[1]),
                float(item[1].get("power", 0.0)),
            ),
            reverse=True,
        )
        return indexed

    def _packing_targets_for_mode(
        self,
        mode: str,
        ordered_templates: list[tuple[int, dict[str, Any]]],
    ) -> list[tuple[float, float]]:
        mode = self._normalize_layout_mode(mode)
        total = len(ordered_templates)
        if total == 0:
            return []

        if mode == "random":
            return [self._sample_random_target(template) for _, template in ordered_templates]

        board_cx, board_cy = self._board_center()
        max_width = max(float(template["width"]) for _, template in ordered_templates)
        max_height = max(float(template["height"]) for _, template in ordered_templates)

        if mode == "cluster":
            anchor_x = float(np.clip(self.rng.normal(board_cx, self.case.outline_width * 0.06), max_width * 0.5, self.case.outline_width - max_width * 0.5))
            anchor_y = float(np.clip(self.rng.normal(board_cy, self.case.outline_height * 0.06), max_height * 0.5, self.case.outline_height - max_height * 0.5))
            base_radius = max(self.min_gap * 4.0, min(self.case.outline_width, self.case.outline_height) * 0.13)
            angle_offset = float(self.rng.uniform(0.0, 2.0 * np.pi))
            targets: list[tuple[float, float]] = []
            for rank, (_, template) in enumerate(ordered_templates):
                fraction = rank / max(total - 1, 1)
                radius = base_radius * (0.15 + 0.85 * fraction)
                theta = angle_offset + 2.0 * np.pi * (0.61803398875 * rank)
                jitter = max(self.min_gap * 0.25, radius * 0.12)
                cx = anchor_x + radius * float(np.cos(theta)) + float(self.rng.normal(0.0, jitter))
                cy = anchor_y + radius * float(np.sin(theta)) + float(self.rng.normal(0.0, jitter))
                targets.append(self._candidate_from_center(template, cx, cy)["cx"],)
            return [
                (
                    float(np.clip(anchor_x + base_radius * (0.15 + 0.85 * (rank / max(total - 1, 1))) * float(np.cos(angle_offset + 2.0 * np.pi * (0.61803398875 * rank))) + float(self.rng.normal(0.0, max(self.min_gap * 0.25, base_radius * 0.12))), 0.0, self.case.outline_width)),
                    float(np.clip(anchor_y + base_radius * (0.15 + 0.85 * (rank / max(total - 1, 1))) * float(np.sin(angle_offset + 2.0 * np.pi * (0.61803398875 * rank))) + float(self.rng.normal(0.0, max(self.min_gap * 0.25, base_radius * 0.12))), 0.0, self.case.outline_height)),
                )
                for rank in range(total)
            ]

        if mode == "boundary":
            edge = str(self.rng.choice(["left", "right", "bottom", "top"]))
            band = max(self.min_gap * 2.0, min(self.case.outline_width, self.case.outline_height) * 0.08)
            step_x = max(max_width + self.min_gap * 2.0, self.case.outline_width / max(2, int(np.ceil(np.sqrt(total)))))
            step_y = max(max_height + self.min_gap * 2.0, self.case.outline_height / max(2, int(np.ceil(np.sqrt(total)))))
            targets: list[tuple[float, float]] = []
            if edge in {"left", "right"}:
                y_positions = np.linspace(band, self.case.outline_height - band, total)
                self.rng.shuffle(y_positions)
                for rank, (_, template) in enumerate(ordered_templates):
                    y = float(np.clip(y_positions[rank] + self.rng.normal(0.0, step_y * 0.08), max_height * 0.5, self.case.outline_height - max_height * 0.5))
                    if rank < max(1, total // 3):
                        x = band if edge == "left" else self.case.outline_width - band
                    else:
                        inward = band * (0.6 + 0.3 * self.rng.random())
                        x = inward if edge == "left" else self.case.outline_width - inward
                    targets.append(self._candidate_from_center(template, x, y)["cx"],)
            else:
                x_positions = np.linspace(band, self.case.outline_width - band, total)
                self.rng.shuffle(x_positions)
                for rank, (_, template) in enumerate(ordered_templates):
                    x = float(np.clip(x_positions[rank] + self.rng.normal(0.0, step_x * 0.08), max_width * 0.5, self.case.outline_width - max_width * 0.5))
                    if rank < max(1, total // 3):
                        y = band if edge == "bottom" else self.case.outline_height - band
                    else:
                        inward = band * (0.6 + 0.3 * self.rng.random())
                        y = inward if edge == "bottom" else self.case.outline_height - inward
                    targets.append(self._candidate_from_center(template, x, y)["cx"],)
            return [
                (
                    float(np.clip(targets[rank][0] if isinstance(targets[rank], tuple) else targets[rank], 0.0, self.case.outline_width)),
                    float(np.clip(targets[rank][1] if isinstance(targets[rank], tuple) else targets[rank], 0.0, self.case.outline_height)),
                )
                for rank in range(len(targets))
            ]

        anchor_count = min(total, max(4, int(np.ceil(np.sqrt(total)))))
        anchors = self._farthest_point_anchors(
            anchor_count,
            0.0,
            self.case.outline_width,
            0.0,
            self.case.outline_height,
            candidate_pool=256,
        )
        if not anchors:
            anchors = [self._board_center()]

        local_step = max(max_width, max_height) + self.min_gap * 4.0
        targets: list[tuple[float, float]] = []
        for rank, (_, template) in enumerate(ordered_templates):
            anchor = anchors[rank % len(anchors)]
            ring = rank // len(anchors)
            angle = 2.0 * np.pi * (0.61803398875 * rank)
            radius = local_step * (0.18 + 0.35 * ring)
            cx = anchor[0] + radius * float(np.cos(angle)) + float(self.rng.normal(0.0, local_step * 0.10))
            cy = anchor[1] + radius * float(np.sin(angle)) + float(self.rng.normal(0.0, local_step * 0.10))
            targets.append(self._candidate_from_center(template, cx, cy)["cx"],)
        return [
            (
                float(np.clip(anchors[rank % len(anchors)][0] + local_step * (0.18 + 0.35 * (rank // len(anchors))) * float(np.cos(2.0 * np.pi * (0.61803398875 * rank))) + float(self.rng.normal(0.0, local_step * 0.10)), 0.0, self.case.outline_width)),
                float(np.clip(anchors[rank % len(anchors)][1] + local_step * (0.18 + 0.35 * (rank // len(anchors))) * float(np.sin(2.0 * np.pi * (0.61803398875 * rank))) + float(self.rng.normal(0.0, local_step * 0.10)), 0.0, self.case.outline_height)),
            )
            for rank in range(total)
        ]

    def _split_free_rect(
        self,
        free_rect: tuple[float, float, float, float],
        placed_rect: tuple[float, float, float, float],
    ) -> list[tuple[float, float, float, float]]:
        x0, y0, x1, y1 = free_rect
        px0, py0, px1, py1 = placed_rect
        pieces: list[tuple[float, float, float, float]] = []
        if px0 > x0:
            pieces.append((x0, y0, px0, y1))
        if px1 < x1:
            pieces.append((px1, y0, x1, y1))
        if py0 > y0:
            pieces.append((px0, y0, px1, py0))
        if py1 < y1:
            pieces.append((px0, py1, px1, y1))
        return pieces

    @staticmethod
    def _rect_area(rect: tuple[float, float, float, float]) -> float:
        return max(0.0, float(rect[2]) - float(rect[0])) * max(0.0, float(rect[3]) - float(rect[1]))

    def _prune_free_rects(self, free_rects: list[tuple[float, float, float, float]]) -> list[tuple[float, float, float, float]]:
        eps = 1e-9
        filtered = [
            rect
            for rect in free_rects
            if float(rect[2]) - float(rect[0]) > eps and float(rect[3]) - float(rect[1]) > eps
        ]
        filtered.sort(key=self._rect_area, reverse=True)
        pruned: list[tuple[float, float, float, float]] = []
        for rect in filtered:
            x0, y0, x1, y1 = rect
            contained = False
            for other in pruned:
                ox0, oy0, ox1, oy1 = other
                if ox0 - eps <= x0 and oy0 - eps <= y0 and ox1 + eps >= x1 and oy1 + eps >= y1:
                    contained = True
                    break
            if not contained:
                pruned.append(rect)
        return pruned

    def _place_in_free_rect(
        self,
        free_rect: tuple[float, float, float, float],
        template: dict[str, Any],
        target: tuple[float, float],
        mode: str,
        placed: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], tuple[float, float, float, float]]:
        x0, y0, x1, y1 = free_rect
        width = float(template["width"])
        height = float(template["height"])
        pad_w = width + self.min_gap
        pad_h = height + self.min_gap
        if x1 - x0 < pad_w or y1 - y0 < pad_h:
            raise ValueError("free rect too small")

        min_px = x0
        max_px = x1 - pad_w
        min_py = y0
        max_py = y1 - pad_h
        desired_px = float(target[0]) - pad_w * 0.5
        desired_py = float(target[1]) - pad_h * 0.5
        px = float(np.clip(desired_px, min_px, max_px))
        py = float(np.clip(desired_py, min_py, max_py))
        actual_x = px + self.min_gap * 0.5
        actual_y = py + self.min_gap * 0.5
        candidate = template.copy()
        candidate["x"] = actual_x
        candidate["y"] = actual_y
        candidate["cx"] = actual_x + width * 0.5
        candidate["cy"] = actual_y + height * 0.5
        candidate = self._clamp_config_position(candidate)

        if not self._is_legal_against(candidate, placed, self.min_gap):
            raise ValueError("candidate violates layout")

        padded_rect = (px, py, px + pad_w, py + pad_h)
        return candidate, padded_rect

    def _select_free_rect_for_target(
        self,
        free_rects: list[tuple[float, float, float, float]],
        template: dict[str, Any],
        target: tuple[float, float],
        mode: str,
        placed: list[dict[str, Any]],
    ) -> tuple[int, dict[str, Any], tuple[float, float, float, float]] | None:
        best: tuple[int, dict[str, Any], tuple[float, float, float, float]] | None = None
        best_score = -float("inf")
        for idx, rect in enumerate(free_rects):
            x0, y0, x1, y1 = rect
            width = float(template["width"])
            height = float(template["height"])
            pad_w = width + self.min_gap
            pad_h = height + self.min_gap
            if x1 - x0 < pad_w or y1 - y0 < pad_h:
                continue
            try:
                candidate, padded_rect = self._place_in_free_rect(rect, template, target, mode, placed)
            except ValueError:
                continue
            center = self._config_center(candidate)
            target_distance = self._distance(center, target)
            pair_distance = self._min_center_distance(candidate, placed)
            edge_distance = self._boundary_distance(candidate)
            if mode == "cluster":
                score = -target_distance + 0.03 * pair_distance - 0.02 * edge_distance
            elif mode == "boundary":
                score = -target_distance - 0.75 * edge_distance + 0.02 * pair_distance
            elif mode == "dispersed":
                score = -0.20 * target_distance + 0.90 * pair_distance + 0.08 * edge_distance
            else:
                score = -0.20 * target_distance + 0.10 * pair_distance
            score += float(self.rng.uniform(0.0, 0.2))
            if score > best_score:
                best = (idx, candidate, padded_rect)
                best_score = score
        return best

    def _generate_packed_topology_layout(
        self,
        mode: str,
        templates: list[dict[str, Any]] | None = None,
        layout_attempts: int = 60,
    ) -> list[dict[str, Any]]:
        mode = self._normalize_layout_mode(mode)
        if templates is None:
            templates = self.base_configs
        templates = [config.copy() for config in templates]
        best_partial: list[dict[str, Any]] = []

        for _ in range(layout_attempts):
            ordered_templates = self._packing_order_for_mode(templates, mode)
            targets = self._packing_targets_for_mode(mode, ordered_templates)
            free_rects: list[tuple[float, float, float, float]] = [
                (0.0, 0.0, float(self.case.outline_width), float(self.case.outline_height))
            ]
            placed: list[dict[str, Any]] = []
            by_index: dict[int, dict[str, Any]] = {}

            for (original_index, template), target in zip(ordered_templates, targets):
                chosen = self._select_free_rect_for_target(free_rects, template, target, mode, placed)
                if chosen is None:
                    break
                rect_index, candidate, padded_rect = chosen
                free_rect = free_rects.pop(rect_index)
                free_rects.extend(self._split_free_rect(free_rect, padded_rect))
                free_rects = self._prune_free_rects(free_rects)
                placed.append(candidate)
                by_index[original_index] = candidate

            if len(placed) > len(best_partial):
                best_partial = placed
            if len(by_index) == len(templates):
                return [by_index[idx] for idx in range(len(templates))]

        raise RuntimeError(
            f"无法生成满足最小间距 {self.min_gap}mm 的 chiplet 布局；"
            f"最多只成功放置 {len(best_partial)}/{len(templates)} 个 chiplet。"
        )

    def _ordered_templates_for_topology(
        self,
        templates: list[dict[str, Any]],
        mode: str,
    ) -> list[tuple[int, dict[str, Any]]]:
        indexed = list(enumerate([config.copy() for config in templates]))
        if mode == "random":
            self.rng.shuffle(indexed)
            return indexed

        if mode in {"cluster", "boundary"}:
            indexed.sort(
                key=lambda item: (
                    float(item[1].get("power", 0.0)),
                    self._area_mm2(item[1]),
                ),
                reverse=True,
            )
            return indexed

        indexed.sort(
            key=lambda item: (
                self._area_mm2(item[1]),
                float(item[1].get("power", 0.0)),
            ),
            reverse=True,
        )
        if len(indexed) <= 2:
            return indexed
        ordered: list[tuple[int, dict[str, Any]]] = []
        left = 0
        right = len(indexed) - 1
        take_left = True
        while left <= right:
            if take_left:
                ordered.append(indexed[left])
                left += 1
            else:
                ordered.append(indexed[right])
                right -= 1
            take_left = not take_left
        return ordered

    def _topology_targets(
        self,
        mode: str,
        ordered_templates: list[tuple[int, dict[str, Any]]],
    ) -> list[tuple[float, float]]:
        mode = self._normalize_layout_mode(mode)
        total = len(ordered_templates)
        if total == 0:
            return []

        if mode == "random":
            return [self._sample_random_target(template) for _, template in ordered_templates]

        max_width = max(float(template["width"]) for _, template in ordered_templates)
        max_height = max(float(template["height"]) for _, template in ordered_templates)
        board_cx, board_cy = self._board_center()

        if mode == "cluster":
            cols = max(2, min(4, int(round(np.sqrt(total)))))
            rows = int(np.ceil(total / cols))
            step_x = max(max_width + self.min_gap * 2.0, min(self.case.outline_width * 0.22, self.case.outline_width / max(cols - 1, 1)))
            step_y = max(max_height + self.min_gap * 2.0, min(self.case.outline_height * 0.22, self.case.outline_height / max(rows - 1, 1)))
            span_x = step_x * max(cols - 1, 1)
            span_y = step_y * max(rows - 1, 1)
            margin_x = max_width * 0.5 + span_x * 0.5 + self.min_gap
            margin_y = max_height * 0.5 + span_y * 0.5 + self.min_gap
            anchor_x = float(np.clip(self.rng.normal(board_cx, self.case.outline_width * 0.06), margin_x, self.case.outline_width - margin_x))
            anchor_y = float(np.clip(self.rng.normal(board_cy, self.case.outline_height * 0.06), margin_y, self.case.outline_height - margin_y))

            cell_offsets: list[tuple[float, float]] = []
            for row in range(rows):
                for col in range(cols):
                    dx = (col - (cols - 1) * 0.5) * step_x
                    dy = (row - (rows - 1) * 0.5) * step_y
                    cell_offsets.append((dx, dy))
            cell_offsets.sort(key=lambda item: item[0] * item[0] + item[1] * item[1])

            jitter_x = max(self.min_gap * 0.5, step_x * 0.08)
            jitter_y = max(self.min_gap * 0.5, step_y * 0.08)
            targets: list[tuple[float, float]] = []
            for rank in range(total):
                dx, dy = cell_offsets[rank % len(cell_offsets)]
                cx = anchor_x + dx + float(self.rng.normal(0.0, jitter_x))
                cy = anchor_y + dy + float(self.rng.normal(0.0, jitter_y))
                targets.append((cx, cy))
            return targets

        if mode == "boundary":
            edge = str(self.rng.choice(["left", "right", "bottom", "top"]))
            if edge in {"left", "right"}:
                cols = max(2, min(4, int(round(np.sqrt(total)))))
                rows = int(np.ceil(total / cols))
                step_x = max(max_width + self.min_gap * 2.0, min(self.case.outline_width * 0.18, self.case.outline_width / max(cols - 1, 1)))
                step_y = max(max_height + self.min_gap * 2.0, min(self.case.outline_height * 0.22, self.case.outline_height / max(rows - 1, 1)))
                span_y = step_y * max(rows - 1, 1)
                margin_y = max_height * 0.5 + span_y * 0.5 + self.min_gap
                anchor_y = float(np.clip(self.rng.normal(board_cy, self.case.outline_height * 0.06), margin_y, self.case.outline_height - margin_y))
                base_x = max_width * 0.5 + self.min_gap if edge == "left" else self.case.outline_width - max_width * 0.5 - self.min_gap
                x_offsets = [col * step_x for col in range(cols)]
                y_offsets = [(row - (rows - 1) * 0.5) * step_y for row in range(rows)]
                cells = [(dx, dy) for dy in y_offsets for dx in x_offsets]
                cells.sort(key=lambda item: (item[0], abs(item[1])))
                jitter_x = max(self.min_gap * 0.5, step_x * 0.05)
                jitter_y = max(self.min_gap * 0.5, step_y * 0.06)
                targets: list[tuple[float, float]] = []
                for rank in range(total):
                    dx, dy = cells[rank % len(cells)]
                    if edge == "right":
                        dx = -dx
                    cx = base_x + dx + float(self.rng.normal(0.0, jitter_x))
                    cy = anchor_y + dy + float(self.rng.normal(0.0, jitter_y))
                    targets.append((cx, cy))
                return targets

            rows = max(2, min(4, int(round(np.sqrt(total)))))
            cols = int(np.ceil(total / rows))
            step_x = max(max_width + self.min_gap * 2.0, min(self.case.outline_width * 0.22, self.case.outline_width / max(cols - 1, 1)))
            step_y = max(max_height + self.min_gap * 2.0, min(self.case.outline_height * 0.18, self.case.outline_height / max(rows - 1, 1)))
            span_x = step_x * max(cols - 1, 1)
            margin_x = max_width * 0.5 + span_x * 0.5 + self.min_gap
            anchor_x = float(np.clip(self.rng.normal(board_cx, self.case.outline_width * 0.06), margin_x, self.case.outline_width - margin_x))
            base_y = max_height * 0.5 + self.min_gap if edge == "bottom" else self.case.outline_height - max_height * 0.5 - self.min_gap
            x_offsets = [(col - (cols - 1) * 0.5) * step_x for col in range(cols)]
            y_offsets = [row * step_y for row in range(rows)]
            cells = [(dx, dy) for dy in y_offsets for dx in x_offsets]
            cells.sort(key=lambda item: (abs(item[0]), item[1]))
            jitter_x = max(self.min_gap * 0.5, step_x * 0.06)
            jitter_y = max(self.min_gap * 0.5, step_y * 0.05)
            targets: list[tuple[float, float]] = []
            for rank in range(total):
                dx, dy = cells[rank % len(cells)]
                if edge == "top":
                    dy = -dy
                cx = anchor_x + dx + float(self.rng.normal(0.0, jitter_x))
                cy = base_y + dy + float(self.rng.normal(0.0, jitter_y))
                targets.append((cx, cy))
            return targets

        anchor_count = min(total, max(4, int(np.ceil(np.sqrt(total)))))
        anchors = self._farthest_point_anchors(
            anchor_count,
            0.0,
            self.case.outline_width,
            0.0,
            self.case.outline_height,
            candidate_pool=256,
        )
        if not anchors:
            anchors = [self._board_center()]

        local_step = max(max_width + self.min_gap * 2.0, max_height + self.min_gap * 2.0, min(self.case.outline_width, self.case.outline_height) * 0.08)
        local_offsets = [
            (0.0, 0.0),
            (local_step, 0.0),
            (-local_step, 0.0),
            (0.0, local_step),
            (0.0, -local_step),
            (local_step * 0.75, local_step * 0.75),
            (-local_step * 0.75, local_step * 0.75),
            (local_step * 0.75, -local_step * 0.75),
            (-local_step * 0.75, -local_step * 0.75),
        ]
        jitter = max(self.min_gap * 0.5, local_step * 0.12)
        targets = []
        for rank in range(total):
            anchor = anchors[rank % len(anchors)]
            ring = rank // len(anchors)
            offset_x, offset_y = local_offsets[ring % len(local_offsets)]
            scale = 1.0 + 0.25 * (ring // len(local_offsets))
            cx = anchor[0] + offset_x * scale + float(self.rng.normal(0.0, jitter))
            cy = anchor[1] + offset_y * scale + float(self.rng.normal(0.0, jitter))
            targets.append((cx, cy))
        return targets

    def _search_topology_candidate(
        self,
        template: dict[str, Any],
        target: tuple[float, float],
        mode: str,
        placed: list[dict[str, Any]],
        candidate_count: int,
    ) -> dict[str, Any] | None:
        mode = self._normalize_layout_mode(mode)
        center_x0, center_y0 = target
        sigma = {
            "random": max(self.min_gap * 4.0, min(self.case.outline_width, self.case.outline_height) * 0.18),
            "cluster": max(self.min_gap * 2.0, min(self.case.outline_width, self.case.outline_height) * 0.07),
            "boundary": max(self.min_gap * 2.0, min(self.case.outline_width, self.case.outline_height) * 0.06),
            "dispersed": max(self.min_gap * 2.0, min(self.case.outline_width, self.case.outline_height) * 0.09),
        }[mode]
        patterns = [
            (0.0, 0.0),
            (1.0, 0.0),
            (-1.0, 0.0),
            (0.0, 1.0),
            (0.0, -1.0),
            (1.0, 1.0),
            (-1.0, 1.0),
            (1.0, -1.0),
            (-1.0, -1.0),
        ]

        best: dict[str, Any] | None = None
        best_score = -float("inf")
        for attempt in range(candidate_count):
            if mode == "random":
                if attempt < candidate_count // 3:
                    dx, dy = patterns[attempt % len(patterns)]
                    scale = sigma * (0.3 + 0.15 * attempt)
                    center_x = center_x0 + dx * scale + float(self.rng.normal(0.0, sigma * 0.25))
                    center_y = center_y0 + dy * scale + float(self.rng.normal(0.0, sigma * 0.25))
                else:
                    center_x = float(self.rng.uniform(self.min_gap, self.case.outline_width - self.min_gap))
                    center_y = float(self.rng.uniform(self.min_gap, self.case.outline_height - self.min_gap))
            else:
                dx, dy = patterns[attempt % len(patterns)]
                scale = sigma * (0.25 + 0.08 * (attempt % 5))
                center_x = center_x0 + dx * scale + float(self.rng.normal(0.0, sigma * 0.35))
                center_y = center_y0 + dy * scale + float(self.rng.normal(0.0, sigma * 0.35))

            candidate = self._candidate_from_center(template, center_x, center_y)
            if not self._is_legal_against(candidate, placed, self.min_gap):
                continue

            center = self._config_center(candidate)
            target_distance = self._distance(center, target)
            edge_distance = self._boundary_distance(candidate)
            clearance = self._clearance_to_layout(candidate, placed)
            pair_distance = self._min_center_distance(candidate, placed)

            if mode == "cluster":
                score = -1.25 * target_distance + 0.05 * clearance - 0.05 * edge_distance
            elif mode == "boundary":
                score = -1.05 * target_distance - 0.80 * edge_distance + 0.05 * clearance
            elif mode == "dispersed":
                score = 1.05 * pair_distance - 0.08 * target_distance + 0.05 * clearance
            else:
                score = -0.18 * target_distance + 0.15 * clearance + 0.04 * pair_distance
            score += float(self.rng.uniform(0.0, 0.25))
            if score > best_score:
                best = candidate
                best_score = score

        return best

    def _generate_topology_layout(
        self,
        mode: str,
        templates: list[dict[str, Any]] | None = None,
        layout_attempts: int = 100,
        candidate_count: int = 256,
    ) -> list[dict[str, Any]]:
        mode = self._normalize_layout_mode(mode)
        if templates is None:
            templates = self.base_configs
        templates = [config.copy() for config in templates]
        best_partial: list[dict[str, Any]] = []

        for _ in range(layout_attempts):
            ordered_templates = self._ordered_templates_for_topology(templates, mode)
            targets = self._topology_targets(mode, ordered_templates)
            placed: list[dict[str, Any]] = []
            by_index: dict[int, dict[str, Any]] = {}

            for (original_index, template), target in zip(ordered_templates, targets):
                candidate = self._search_topology_candidate(
                    template,
                    target,
                    mode,
                    placed,
                    candidate_count=candidate_count,
                )
                if candidate is None:
                    break
                placed.append(candidate)
                by_index[original_index] = candidate

            if len(placed) > len(best_partial):
                best_partial = placed
            if len(by_index) == len(templates):
                return [by_index[idx] for idx in range(len(templates))]

        raise RuntimeError(
            f"无法生成满足最小间距 {self.min_gap}mm 的 chiplet 布局；"
            f"最多只成功放置 {len(best_partial)}/{len(templates)} 个 chiplet。"
        )

    def _set_config_center(self, config: dict[str, Any], center_x: float, center_y: float) -> dict[str, Any]:
        config["cx"] = float(center_x)
        config["cy"] = float(center_y)
        config["x"] = float(center_x) - float(config["width"]) * 0.5
        config["y"] = float(center_y) - float(config["height"]) * 0.5
        return self._clamp_config_position(config)

    def _rotate_config(self, config: dict[str, Any], rotation: int) -> dict[str, Any]:
        rotation = 90 if int(rotation) == 90 else 0
        center_x, center_y = self._config_center(config)
        rotated = config.copy()
        base_width = float(rotated.get("base_width", rotated["width"]))
        base_height = float(rotated.get("base_height", rotated["height"]))
        if rotation == 90:
            rotated["width"] = base_height
            rotated["height"] = base_width
        else:
            rotated["width"] = base_width
            rotated["height"] = base_height
        rotated["rotation"] = rotation
        return self._set_config_center(rotated, center_x, center_y)

    def _apply_random_rotation(self, configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [self._rotate_config(config, int(self.rng.choice([0, 90]))) for config in configs]

    def _power_bounds_for_config(self, config: dict[str, Any]) -> tuple[float, float]:
        area = self._area_mm2(config)
        min_power = max(0.0, self.min_power_density * area)
        max_power = max(min_power, float(self.max_power_density) * area)
        return min_power, max_power

    def _apply_power_tdp_limit(self, configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        total_power = float(sum(config["power"] for config in configs))
        if total_power <= self.default_tdp_limit or total_power <= 0.0:
            return configs
        scale = self.default_tdp_limit / total_power
        for config in configs:
            min_power, max_power = self._power_bounds_for_config(config)
            config["power"] = float(np.clip(float(config["power"]) * scale, min_power, max_power))
        return configs

    def _apply_power_variation(self, configs: list[dict[str, Any]], power_range: tuple[float, float]) -> list[dict[str, Any]]:
        additive_scale = self.mean_base_power * self.power_additive_fraction
        varied: list[dict[str, Any]] = []
        for config in configs:
            new_config = config.copy()
            base_power = float(new_config.get("base_power", new_config["power"]))
            draw = float(self.rng.random())
            if draw < self.power_shutdown_prob:
                power = 0.0
            elif draw < self.power_shutdown_prob + self.power_dropout_prob:
                power = base_power * self.power_sleep_ratio
            else:
                power_mult = float(self.rng.uniform(*power_range))
                additive = float(self.rng.uniform(-additive_scale, additive_scale))
                power = base_power * power_mult + additive
            min_power, max_power = self._power_bounds_for_config(new_config)
            new_config["power"] = float(np.clip(power, min_power, max_power))
            varied.append(new_config)
        return self._apply_power_tdp_limit(varied)

    def _chiplet_bounds(self, config: dict[str, Any], gap: float = 0.0) -> tuple[float, float, float, float]:
        return (
            float(config["x"]) - gap,
            float(config["x"]) + float(config["width"]) + gap,
            float(config["y"]) - gap,
            float(config["y"]) + float(config["height"]) + gap,
        )

    def _is_position_in_bounds(self, config: dict[str, Any]) -> bool:
        return (
            float(config["width"]) <= self.case.outline_width
            and float(config["height"]) <= self.case.outline_height
            and 0.0 <= float(config["x"]) <= self.case.outline_width - float(config["width"])
            and 0.0 <= float(config["y"]) <= self.case.outline_height - float(config["height"])
        )

    def _has_pair_conflict(self, first: dict[str, Any], second: dict[str, Any], min_gap: float) -> bool:
        first_left, first_right, first_bottom, first_top = self._chiplet_bounds(first)
        second_left, second_right, second_bottom, second_top = self._chiplet_bounds(second)
        return not (
            first_right + min_gap <= second_left
            or second_right + min_gap <= first_left
            or first_top + min_gap <= second_bottom
            or second_top + min_gap <= first_bottom
        )

    def _is_legal_against(self, config: dict[str, Any], placed: list[dict[str, Any]], min_gap: float) -> bool:
        return self._is_position_in_bounds(config) and all(
            not self._has_pair_conflict(config, other, min_gap) for other in placed
        )

    def _clearance_to_layout(self, config: dict[str, Any], placed: list[dict[str, Any]]) -> float:
        edge_clearance = min(
            float(config["x"]),
            float(config["y"]),
            self.case.outline_width - (float(config["x"]) + float(config["width"])),
            self.case.outline_height - (float(config["y"]) + float(config["height"])),
        )
        if not placed:
            return float(edge_clearance)

        clearance = float(edge_clearance)
        left, right, bottom, top = self._chiplet_bounds(config)
        for other in placed:
            other_left, other_right, other_bottom, other_top = self._chiplet_bounds(other)
            x_sep = max(other_left - right, left - other_right, 0.0)
            y_sep = max(other_bottom - top, bottom - other_top, 0.0)
            if x_sep > 0.0 or y_sep > 0.0:
                clearance = min(clearance, max(x_sep, y_sep))
            else:
                return -float("inf")
        return clearance

    def _check_chiplet_collision(self, configs: list[dict[str, Any]], min_gap: float) -> tuple[bool, str]:
        for config in configs:
            if not self._is_position_in_bounds(config):
                return False, f"Chiplet {config['name']} 超出 interposer 范围"
        for i in range(len(configs)):
            for j in range(i + 1, len(configs)):
                if self._has_pair_conflict(configs[i], configs[j], min_gap):
                    return False, f"Chiplet {configs[i]['name']} 和 {configs[j]['name']} 相交或间距不足"
        return True, ""

    def _sample_candidate_position(self, template: dict[str, Any], mode: str, pos_range: float) -> dict[str, Any] | None:
        max_x = self.case.outline_width - float(template["width"])
        max_y = self.case.outline_height - float(template["height"])
        if max_x < 0.0 or max_y < 0.0:
            return None

        candidate = template.copy()
        if mode == "local":
            candidate["x"] = float(template["x"]) + float(self.rng.uniform(-pos_range, pos_range))
            candidate["y"] = float(template["y"]) + float(self.rng.uniform(-pos_range, pos_range))
        elif mode == "edge":
            edge = str(self.rng.choice(["left", "right", "bottom", "top"]))
            if edge == "left":
                candidate["x"] = float(self.rng.uniform(0.0, max_x * 0.25))
                candidate["y"] = float(self.rng.uniform(0.0, max_y))
            elif edge == "right":
                candidate["x"] = float(self.rng.uniform(max_x * 0.75, max_x))
                candidate["y"] = float(self.rng.uniform(0.0, max_y))
            elif edge == "bottom":
                candidate["x"] = float(self.rng.uniform(0.0, max_x))
                candidate["y"] = float(self.rng.uniform(0.0, max_y * 0.25))
            else:
                candidate["x"] = float(self.rng.uniform(0.0, max_x))
                candidate["y"] = float(self.rng.uniform(max_y * 0.75, max_y))
        elif mode == "stratified":
            grid = max(2, int(np.ceil(np.sqrt(len(self.base_configs)))))
            cell_x = int(self.rng.integers(0, grid))
            cell_y = int(self.rng.integers(0, grid))
            candidate["x"] = float(self.rng.uniform(cell_x / grid * max_x, (cell_x + 1) / grid * max_x))
            candidate["y"] = float(self.rng.uniform(cell_y / grid * max_y, (cell_y + 1) / grid * max_y))
        else:
            candidate["x"] = float(self.rng.uniform(0.0, max_x))
            candidate["y"] = float(self.rng.uniform(0.0, max_y))

        return self._clamp_config_position(candidate)

    def _find_legal_position(
        self,
        template: dict[str, Any],
        placed: list[dict[str, Any]],
        min_gap: float,
        pos_range: float,
        candidate_count: int = 256,
        prefer_jump: bool = False,
    ) -> dict[str, Any] | None:
        best: dict[str, Any] | None = None
        best_score = -float("inf")
        modes = ["uniform", "uniform", "stratified", "edge", "local"] if prefer_jump else ["local", "uniform", "stratified", "edge"]

        for _ in range(candidate_count):
            mode = str(self.rng.choice(modes))
            candidate = self._sample_candidate_position(template, mode, pos_range)
            if candidate is None or not self._is_legal_against(candidate, placed, min_gap):
                continue
            if placed:
                clearance = self._clearance_to_layout(candidate, placed)
                base_dx = abs(float(candidate["x"]) - float(template["x"]))
                base_dy = abs(float(candidate["y"]) - float(template["y"]))
                jump_bonus = 0.1 * (base_dx + base_dy) if prefer_jump else 0.0
                score = clearance + jump_bonus
            else:
                score = 0.0
                if mode in ("edge", "stratified"):
                    score += max(1.0, min_gap)
            score += float(self.rng.uniform(0.0, max(1.0, min_gap)))
            if score > best_score:
                best = candidate
                best_score = score
        return best

    def _generate_legal_layout(
        self,
        pos_range: float,
        min_gap: float,
        templates: list[dict[str, Any]] | None = None,
        layout_attempts: int = 80,
    ) -> list[dict[str, Any]]:
        if templates is None:
            templates = self.base_configs
        indexed_configs = list(enumerate(templates))
        indexed_configs.sort(key=lambda item: item[1]["width"] * item[1]["height"], reverse=True)

        best_partial: list[dict[str, Any]] = []
        for attempt in range(layout_attempts):
            order = indexed_configs.copy()
            if attempt > 0:
                groups: dict[int, list[tuple[int, dict[str, Any]]]] = {}
                for item in order:
                    area_key = int(round(item[1]["width"] * item[1]["height"], -4))
                    groups.setdefault(area_key, []).append(item)
                order = []
                for key in sorted(groups.keys(), reverse=True):
                    group = groups[key]
                    self.rng.shuffle(group)
                    order.extend(group)

            placed: list[dict[str, Any]] = []
            by_index: dict[int, dict[str, Any]] = {}
            for original_index, template in order:
                prefer_jump = attempt > layout_attempts // 3 or float(self.rng.random()) < 0.45
                candidate = self._find_legal_position(
                    template,
                    placed,
                    min_gap,
                    pos_range,
                    candidate_count=192,
                    prefer_jump=prefer_jump,
                )
                if candidate is None:
                    break
                placed.append(candidate)
                by_index[original_index] = candidate

            if len(placed) > len(best_partial):
                best_partial = placed
            if len(by_index) == len(templates):
                return [by_index[idx] for idx in range(len(templates))]

        raise RuntimeError(
            f"无法生成满足最小间距 {min_gap}mm 的 chiplet 布局；"
            f"最多只成功放置 {len(best_partial)}/{len(templates)} 个 chiplet。"
        )

    def _area_similarity(self, first: dict[str, Any], second: dict[str, Any]) -> float:
        first_area = float(first["width"]) * float(first["height"])
        second_area = float(second["width"]) * float(second["height"])
        return min(first_area, second_area) / max(first_area, second_area)

    def _position_from_center(self, template: dict[str, Any], center_x: float, center_y: float) -> dict[str, Any]:
        candidate = template.copy()
        candidate["x"] = float(center_x) - float(candidate["width"]) * 0.5
        candidate["y"] = float(center_y) - float(candidate["height"]) * 0.5
        return self._clamp_config_position(candidate)

    def _try_swap_mutation(
        self,
        configs: list[dict[str, Any]],
        min_gap: float,
        pos_range: float,
        area_similarity_threshold: float = 0.65,
        candidate_count: int = 48,
    ) -> bool:
        if len(configs) < 2:
            return False

        idx_a, idx_b = self.rng.choice(len(configs), size=2, replace=False)
        idx_a = int(idx_a)
        idx_b = int(idx_b)
        chip_a = configs[idx_a]
        chip_b = configs[idx_b]
        similar_area = self._area_similarity(chip_a, chip_b) >= area_similarity_threshold
        center_a = self._config_center(chip_a)
        center_b = self._config_center(chip_b)
        others = [config for idx, config in enumerate(configs) if idx not in (idx_a, idx_b)]
        jitter_range = min(pos_range * 0.15, max(min_gap * 4.0, 1.0))

        best_pair: tuple[dict[str, Any], dict[str, Any]] | None = None
        best_score = -float("inf")
        for attempt in range(candidate_count):
            if attempt == 0:
                jitter_a = (0.0, 0.0)
                jitter_b = (0.0, 0.0)
            else:
                jitter_a = (
                    float(self.rng.uniform(-jitter_range, jitter_range)),
                    float(self.rng.uniform(-jitter_range, jitter_range)),
                )
                jitter_b = (
                    float(self.rng.uniform(-jitter_range, jitter_range)),
                    float(self.rng.uniform(-jitter_range, jitter_range)),
                )

            new_a = self._position_from_center(chip_a, center_b[0] + jitter_a[0], center_b[1] + jitter_a[1])
            new_b = self._position_from_center(chip_b, center_a[0] + jitter_b[0], center_a[1] + jitter_b[1])

            if self._has_pair_conflict(new_a, new_b, min_gap):
                continue
            if not self._is_legal_against(new_a, others, min_gap):
                continue
            if not self._is_legal_against(new_b, others + [new_a], min_gap):
                continue

            clearance = min(
                self._clearance_to_layout(new_a, others + [new_b]),
                self._clearance_to_layout(new_b, others + [new_a]),
            )
            score = clearance + (min_gap if similar_area else 0.0)
            score += float(self.rng.uniform(0.0, max(1.0, min_gap)))
            if score > best_score:
                best_pair = (new_a, new_b)
                best_score = score

        if best_pair is None:
            return False

        configs[idx_a] = best_pair[0]
        configs[idx_b] = best_pair[1]
        return True

    def _apply_position_jumps(
        self,
        configs: list[dict[str, Any]],
        pos_range: float,
        min_gap: float,
        jump_probability: float = 0.35,
        swap_probability: float = 0.30,
        mutation_rounds: int | None = None,
    ) -> list[dict[str, Any]]:
        if mutation_rounds is None:
            mutation_rounds = max(4, len(configs) * 2)

        configs = [config.copy() for config in configs]
        for _ in range(mutation_rounds):
            if float(self.rng.random()) < swap_probability:
                if self._try_swap_mutation(configs, min_gap, pos_range):
                    continue

            idx = int(self.rng.integers(0, len(configs)))
            current = configs[idx]
            others = configs[:idx] + configs[idx + 1 :]
            prefer_jump = float(self.rng.random()) < jump_probability
            candidate = self._find_legal_position(
                current,
                others,
                min_gap,
                pos_range,
                candidate_count=128,
                prefer_jump=prefer_jump,
            )
            if candidate is not None:
                configs[idx] = candidate
        return configs

    def _generate_random_variation(
        self,
        pos_range: float = 2.0,
        power_range: tuple[float, float] = (0.5, 1.5),
    ) -> list[dict[str, Any]]:
        return self._generate_layout_variation(self.layout_mode, pos_range=pos_range, power_range=power_range)

    def _generate_fallback_layout(self, templates: list[dict[str, Any]], min_gap: float) -> list[dict[str, Any]]:
        try:
            return self._generate_legal_layout(
                pos_range=max(min_gap * 4.0, min(self.case.outline_width, self.case.outline_height) * 0.12),
                min_gap=min_gap,
                templates=templates,
                layout_attempts=max(80, len(templates) * 4),
            )
        except RuntimeError:
            pass

        ordered = sorted(
            [config.copy() for config in templates],
            key=lambda item: (float(item["width"]) * float(item["height"]), float(item["width"])),
            reverse=True,
        )
        placements: list[dict[str, Any]] = []
        cursor_x = 0.0
        cursor_y = 0.0
        row_height = 0.0
        fallback_gap = max(min_gap, 0.0)

        for config in ordered:
            width = float(config["width"])
            height = float(config["height"])
            if cursor_x > 0.0 and cursor_x + width > self.case.outline_width:
                cursor_x = 0.0
                cursor_y += row_height + fallback_gap
                row_height = 0.0
            if cursor_y + height > self.case.outline_height:
                raise RuntimeError(
                    f"无法生成满足最小间距 {min_gap}mm 的 chiplet 布局；fallback shelf packing 仍然失败"
                )
            config["x"] = cursor_x
            config["y"] = cursor_y
            config["cx"] = cursor_x + width * 0.5
            config["cy"] = cursor_y + height * 0.5
            placements.append(config)
            cursor_x += width + fallback_gap
            row_height = max(row_height, height)

        if len(placements) != len(templates):
            raise RuntimeError("fallback shelf packing did not place all chiplets")
        return placements

    def _generate_grid_variation(self, sample_id: int, total_samples: int) -> list[dict[str, Any]]:
        configs: list[dict[str, Any]] = []
        power_mult = 0.5 + (sample_id / max(total_samples, 1)) * 1.0
        for config in self.base_configs:
            new_config = config.copy()
            new_config["power"] = float(new_config["power"]) * power_mult
            new_config = self._clamp_config_position(new_config)
            configs.append(new_config)
        return configs

    def _build_layout(self, configs: list[dict[str, Any]]) -> Layout:
        placements = tuple(
            Placement(
                chiplet_id=str(config["name"]),
                x=float(config.get("cx", float(config["x"]) + float(config["width"]) * 0.5)),
                y=float(config.get("cy", float(config["y"]) + float(config["height"]) * 0.5)),
                rotation=int(config.get("rotation", 0)) % 360,
            )
            for config in configs
        )
        return Layout(placements)

    def _chiplet_at_point(self, grid_x: float, grid_y: float, configs: list[dict[str, Any]]) -> tuple[str, float]:
        eps = 1e-9
        for config in configs:
            x0 = float(config["x"])
            y0 = float(config["y"])
            x1 = x0 + float(config["width"])
            y1 = y0 + float(config["height"])
            if x0 - eps <= grid_x <= x1 + eps and y0 - eps <= grid_y <= y1 + eps:
                return str(config.get("name", "background")), float(config["power"])
        return "background", 0.0

    def generate_sample(
        self,
        sample_id: int,
        chiplet_configs: list[dict[str, Any]],
        *,
        variation_type: str = "random",
    ) -> ThermalSample | None:
        configs = [self._clamp_config_position(config.copy()) for config in chiplet_configs]
        valid, message = self._check_chiplet_collision(configs, self.min_gap)
        if not valid:
            print(f"[error] sample {sample_id} illegal layout: {message}")
            return None

        layout = self._build_layout(configs)
        try:
            temperature_map = self.backend.simulate(self.case, layout)
        except Exception as exc:
            print(f"[error] sample {sample_id} thermal simulation failed: {exc}")
            return None

        return ThermalSample(
            sample_id=sample_id,
            timestamp=datetime.now().isoformat(),
            variation_type=str(variation_type),
            layout_mode=self.layout_mode,
            chiplet_configs=configs,
            temperature_map=temperature_map,
            temperature_stats={
                "min": float(np.min(temperature_map)),
                "max": float(np.max(temperature_map)),
                "mean": float(np.mean(temperature_map)),
                "std": float(np.std(temperature_map)),
            },
        )

    def save_sample_format_pointwise(self, sample_data: ThermalSample, output_file: Path) -> None:
        temp_map = sample_data.temperature_map
        configs = sample_data.chiplet_configs
        rows, cols = temp_map.shape
        with output_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["grid_x", "grid_y", "chiplet_id", "chiplet_power", "temperature"])
            for row_idx in range(rows):
                for col_idx in range(cols):
                    grid_x = float(col_idx) / max(cols - 1, 1) * self.case.outline_width
                    grid_y = float(row_idx) / max(rows - 1, 1) * self.case.outline_height
                    chiplet_id, power = self._chiplet_at_point(grid_x, grid_y, configs)
                    writer.writerow(
                        [
                            f"{grid_x:.6f}",
                            f"{grid_y:.6f}",
                            chiplet_id,
                            f"{power:.6f}",
                            f"{temp_map[row_idx, col_idx]:.6f}",
                        ]
                    )

    def save_sample_format_pointwise_augmented(self, sample_data: ThermalSample, output_file: Path) -> None:
        temp_map = sample_data.temperature_map
        configs = sample_data.chiplet_configs
        rows, cols = temp_map.shape
        total = rows * cols
        grid_x = np.empty(total, dtype=np.float32)
        grid_y = np.empty(total, dtype=np.float32)
        chiplet_ids = np.empty(total, dtype=object)
        chiplet_power = np.empty(total, dtype=np.float32)
        temperatures = np.empty(total, dtype=np.float32)

        idx = 0
        for row_idx in range(rows):
            for col_idx in range(cols):
                gx = float(col_idx) / max(cols - 1, 1) * self.case.outline_width
                gy = float(row_idx) / max(rows - 1, 1) * self.case.outline_height
                chiplet_id, power = self._chiplet_at_point(gx, gy, configs)
                grid_x[idx] = gx
                grid_y[idx] = gy
                chiplet_ids[idx] = chiplet_id
                chiplet_power[idx] = power
                temperatures[idx] = float(temp_map[row_idx, col_idx])
                idx += 1

        payload = {
            "chiplet_configs": configs,
            "system_config": {
                "interposer_width": self.case.outline_width,
                "interposer_height": self.case.outline_height,
            },
        }
        features = build_pointwise_feature_columns(grid_x, grid_y, payload, chiplet_ids=chiplet_ids)

        fieldnames = [
            "grid_x",
            "grid_y",
            "chiplet_id",
            "chiplet_power",
            "temperature",
            "occupancy_mask",
            "edge_mask",
            "coord_x_norm",
            "coord_y_norm",
        ]

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="raise")
            writer.writeheader()
            for index in range(total):
                writer.writerow(
                    {
                        "grid_x": f"{float(grid_x[index]):.6f}",
                        "grid_y": f"{float(grid_y[index]):.6f}",
                        "chiplet_id": str(chiplet_ids[index]),
                        "chiplet_power": f"{float(chiplet_power[index]):.6f}",
                        "temperature": f"{float(temperatures[index]):.6f}",
                        "occupancy_mask": int(features["occupancy_mask"][index]),
                        "edge_mask": int(features["edge_mask"][index]),
                        "coord_x_norm": f"{float(features['coord_x_norm'][index]):.6f}",
                        "coord_y_norm": f"{float(features['coord_y_norm'][index]):.6f}",
                    }
                )

    def save_sample_format_gridwise(self, sample_data: ThermalSample, output_file: Path) -> None:
        temp_map = sample_data.temperature_map
        with output_file.open("w", newline="", encoding="utf-8") as f:
            f.write(f"# Sample ID: {sample_data.sample_id}\n")
            f.write(
                f"# Temperature Stats: min={sample_data.temperature_stats['min']:.2f}, "
                f"max={sample_data.temperature_stats['max']:.2f}, "
                f"mean={sample_data.temperature_stats['mean']:.2f}\n"
            )
            f.write(f"# Chiplets: {len(sample_data.chiplet_configs)}\n")
            writer = csv.writer(f)
            for row in temp_map:
                writer.writerow([f"{value:.6f}" for value in row])

    def save_sample_format_json(self, sample_data: ThermalSample, output_file: Path) -> None:
        data_to_save = {
            "sample_id": sample_data.sample_id,
            "timestamp": sample_data.timestamp,
            "variation_type": sample_data.variation_type,
            "layout_mode": sample_data.layout_mode,
            "chiplet_configs": sample_data.chiplet_configs,
            "temperature_map": sample_data.temperature_map.tolist(),
            "temperature_stats": sample_data.temperature_stats,
            "system_config": {
                "interposer_width": self.case.outline_width,
                "interposer_height": self.case.outline_height,
                "num_grid_x": int(sample_data.temperature_map.shape[1]),
                "num_grid_y": int(sample_data.temperature_map.shape[0]),
            },
            "thermal_backend": {
                "requested": str(self.thermal_config.get("backend", "hotspot")),
                "runtime_mode": getattr(self.backend, "runtime_mode", self.backend.name),
            },
        }
        output_file.write_text(json.dumps(data_to_save, indent=2, ensure_ascii=False), encoding="utf-8")

    def generate_dataset(
        self,
        num_samples: int,
        output_dir: Path | str,
        variation_type: str = "random",
        save_formats: list[str] | None = None,
    ) -> Path:
        if save_formats is None:
            save_formats = ["pointwise", "json"]

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for fmt in save_formats:
            (output_dir / fmt).mkdir(exist_ok=True)

        successful = 0
        failed = 0
        attempted = 0
        target_successes = max(0, int(num_samples))
        while successful < target_successes:
            try:
                if variation_type == "fixed":
                    configs = [config.copy() for config in self.base_configs]
                elif variation_type == "random":
                    configs = self._generate_layout_variation(self.layout_mode)
                elif variation_type == "grid":
                    configs = self._generate_grid_variation(attempted, target_successes)
                else:
                    raise ValueError(f"unknown variation_type: {variation_type}")
            except Exception as exc:
                failed += 1
                print(f"[warn] sample attempt {attempted} layout generation failed: {exc}")
                attempted += 1
                continue

            sample_id = successful
            sample_data = self.generate_sample(sample_id, configs, variation_type=variation_type)
            if sample_data is None:
                failed += 1
                attempted += 1
                if variation_type == "fixed":
                    raise RuntimeError("fixed variation failed to generate a valid sample")
                continue

            for fmt in save_formats:
                if fmt == "pointwise":
                    self.save_sample_format_pointwise(sample_data, output_dir / "pointwise" / f"sample_{sample_id:06d}.csv")
                elif fmt == "pointwise_augmented":
                    self.save_sample_format_pointwise_augmented(
                        sample_data,
                        output_dir / "pointwise_augmented" / f"sample_{sample_id:06d}.csv",
                    )
                elif fmt == "gridwise":
                    self.save_sample_format_gridwise(sample_data, output_dir / "gridwise" / f"sample_{sample_id:06d}.csv")
                elif fmt == "json":
                    self.save_sample_format_json(sample_data, output_dir / "json" / f"sample_{sample_id:06d}.json")
                else:
                    raise ValueError(f"unknown save format: {fmt}")

            successful += 1
            attempted += 1

        self._save_dataset_summary(output_dir, target_successes, attempted, successful, failed)
        return output_dir

    def _save_dataset_summary(self, output_dir: Path, requested_samples: int, attempted: int, successful: int, failed: int) -> None:
        summary = {
            "generation_time": datetime.now().isoformat(),
            "case": self.case_dir.name,
            "requested_samples": requested_samples,
            "attempted_samples": attempted,
            "total_samples": successful,
            "successful_samples": successful,
            "failed_samples": failed,
            "success_rate": successful / attempted if attempted > 0 else 0.0,
            "system_config": {
                "interposer_width": self.case.outline_width,
                "interposer_height": self.case.outline_height,
                "num_grid_x": int(self.thermal_config.get("grid_size", [64, 64])[0]),
                "num_grid_y": int(self.thermal_config.get("grid_size", [64, 64])[1]),
                "num_chiplets": len(self.base_configs),
            },
            "thermal_config": self.thermal_config,
            "thermal_backend": {
                "requested": str(self.thermal_config.get("backend", "hotspot")),
                "runtime_mode": getattr(self.backend, "runtime_mode", self.backend.name),
            },
            "layout_mode": self.layout_mode,
            "randomization_config": {
                "randomize_position": self.randomize_position,
                "randomize_power": self.randomize_power,
                "effective_randomize_power": self.randomize_power and self.layout_mode not in {"cluster", "boundary"},
                "randomize_rotation": self.randomize_rotation,
                "min_gap": self.min_gap,
                "power_additive_fraction": self.power_additive_fraction,
                "power_dropout_prob": self.power_dropout_prob,
                "power_sleep_ratio": self.power_sleep_ratio,
                "power_shutdown_prob": self.power_shutdown_prob,
                "min_power_density": self.min_power_density,
                "max_power_density": self.max_power_density,
                "tdp_limit": self.default_tdp_limit,
                "base_total_power": self.base_total_power,
            },
            "base_configurations": self.base_configs,
        }
        (output_dir / "dataset_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def dry_run_generation(
        self,
        num_samples: int = 1,
        variation_type: str = "random",
        layout_modes: list[str] | None = None,
    ) -> dict[str, Any]:
        modes = [self.layout_mode] if layout_modes is None else [self._normalize_layout_mode(mode) for mode in layout_modes]
        results: list[dict[str, Any]] = []
        target = max(0, int(num_samples))
        for mode in modes:
            for sample_index in range(target):
                if variation_type == "fixed":
                    configs = [config.copy() for config in self.base_configs]
                elif variation_type == "grid":
                    configs = self._generate_grid_variation(sample_index, max(1, target))
                elif variation_type == "random":
                    configs = self._generate_layout_variation(mode)
                else:
                    raise ValueError(f"unknown variation_type: {variation_type}")

                valid, message = self._check_chiplet_collision(configs, self.min_gap)
                if not valid:
                    raise RuntimeError(f"dry-run layout invalid for mode={mode}: {message}")
                layout = self._build_layout(configs)
                temperature_map = self.backend.simulate(self.case, layout)
                results.append(
                    {
                        "mode": mode,
                        "sample_id": sample_index,
                        "chiplets": len(configs),
                        "temperature_stats": {
                            "min": float(np.min(temperature_map)),
                            "max": float(np.max(temperature_map)),
                            "mean": float(np.mean(temperature_map)),
                            "std": float(np.std(temperature_map)),
                        },
                    }
                )
        return {
            "case": self.case_dir.name,
            "variation_type": variation_type,
            "min_gap": self.min_gap,
            "layout_modes": modes,
            "requested_samples_per_mode": target,
            "results": results,
        }
