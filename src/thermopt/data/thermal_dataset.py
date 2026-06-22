from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from thermopt.data.atplace import load_atplace_case
from thermopt.layout.objects import FloorplanCase, Layout, Placement
from thermopt.thermal.backend import ThermalBackend, build_thermal_backend


@dataclass(frozen=True)
class ThermalSample:
    sample_id: int
    timestamp: str
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
        self.backend: ThermalBackend = build_thermal_backend(self.case, self.thermal_config, work_dir=work_dir)

    @staticmethod
    def _layout_to_configs(case: FloorplanCase, layout: Layout) -> list[dict[str, Any]]:
        chiplets = case.chiplet_by_id
        configs: list[dict[str, Any]] = []
        for placement in layout.placements:
            chiplet = chiplets[placement.chiplet_id]
            configs.append(
                {
                    "name": placement.chiplet_id,
                    "x": float(placement.x),
                    "y": float(placement.y),
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
        return config

    @staticmethod
    def _area_mm2(config: dict[str, Any]) -> float:
        return float(config["width"]) * float(config["height"])

    @staticmethod
    def _config_center(config: dict[str, Any]) -> tuple[float, float]:
        return (
            float(config["x"]) + float(config["width"]) * 0.5,
            float(config["y"]) + float(config["height"]) * 0.5,
        )

    def _set_config_center(self, config: dict[str, Any], center_x: float, center_y: float) -> dict[str, Any]:
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
        configs = [config.copy() for config in self.base_configs]

        if self.randomize_rotation:
            configs = self._apply_random_rotation(configs)
        else:
            configs = [self._rotate_config(config, config.get("rotation", 0)) for config in configs]

        if self.randomize_position:
            configs = self._generate_legal_layout(pos_range, self.min_gap, templates=configs)
            configs = self._apply_position_jumps(configs, pos_range, self.min_gap)
        else:
            configs = [self._clamp_config_position(config) for config in configs]

        if self.randomize_power:
            configs = self._apply_power_variation(configs, power_range)

        valid, message = self._check_chiplet_collision(configs, self.min_gap)
        if not valid:
            raise RuntimeError(f"随机布局生成失败: {message}")
        return configs

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
                x=float(config["x"]),
                y=float(config["y"]),
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

    def generate_sample(self, sample_id: int, chiplet_configs: list[dict[str, Any]]) -> ThermalSample | None:
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
                    configs = self._generate_random_variation()
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
            sample_data = self.generate_sample(sample_id, configs)
            if sample_data is None:
                failed += 1
                attempted += 1
                if variation_type == "fixed":
                    raise RuntimeError("fixed variation failed to generate a valid sample")
                continue

            for fmt in save_formats:
                if fmt == "pointwise":
                    self.save_sample_format_pointwise(sample_data, output_dir / "pointwise" / f"sample_{sample_id:06d}.csv")
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
            "randomization_config": {
                "randomize_position": self.randomize_position,
                "randomize_power": self.randomize_power,
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
