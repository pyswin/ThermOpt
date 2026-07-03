from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PointwiseFeatureConfig:
    edge_band_mm: float | None = None


def _load_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows, fieldnames


def _load_sample_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if "chiplet_configs" not in payload:
        raise ValueError(f"{path} is missing chiplet_configs")
    return payload


def _extract_outline(payload: dict[str, Any], grid_x: np.ndarray, grid_y: np.ndarray) -> tuple[float, float]:
    system_config = payload.get("system_config", {})
    outline_width = float(system_config.get("interposer_width", float(np.max(grid_x))))
    outline_height = float(system_config.get("interposer_height", float(np.max(grid_y))))
    return outline_width, outline_height


def _grid_spacing(values: np.ndarray) -> float:
    unique = np.unique(np.asarray(values, dtype=np.float32))
    if unique.size < 2:
        return 1.0
    diffs = np.diff(np.sort(unique))
    positive = diffs[diffs > 1e-9]
    if positive.size == 0:
        return 1.0
    return float(np.median(positive))


def _default_edge_band_mm(grid_x: np.ndarray, grid_y: np.ndarray) -> float:
    return 0.5 * min(_grid_spacing(grid_x), _grid_spacing(grid_y))


def _distance_to_rectangle_boundary(
    x: np.ndarray,
    y: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> np.ndarray:
    inside = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    distance = np.full_like(x, np.inf, dtype=np.float32)

    if np.any(inside):
        interior = np.minimum.reduce(
            [
                x[inside] - x0,
                x1 - x[inside],
                y[inside] - y0,
                y1 - y[inside],
            ]
        ).astype(np.float32)
        distance[inside] = interior

    if np.any(~inside):
        dx = np.where(x < x0, x0 - x, np.where(x > x1, x - x1, 0.0)).astype(np.float32)
        dy = np.where(y < y0, y0 - y, np.where(y > y1, y - y1, 0.0)).astype(np.float32)
        distance[~inside] = np.hypot(dx[~inside], dy[~inside]).astype(np.float32)

    return distance


def build_pointwise_feature_columns(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    payload: dict[str, Any],
    *,
    edge_band_mm: float | None = None,
    chiplet_ids: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    grid_x = np.asarray(grid_x, dtype=np.float32)
    grid_y = np.asarray(grid_y, dtype=np.float32)
    if grid_x.shape != grid_y.shape:
        raise ValueError("grid_x and grid_y must have the same shape")

    outline_width, outline_height = _extract_outline(payload, grid_x, grid_y)
    chiplets = list(payload.get("chiplet_configs", []))

    if chiplet_ids is not None:
        chiplet_ids = np.asarray(chiplet_ids)
        if chiplet_ids.shape != grid_x.shape:
            raise ValueError("chiplet_ids must have the same shape as grid_x and grid_y")
        occupancy = (chiplet_ids != "background").astype(np.uint8)
    else:
        occupancy = np.zeros_like(grid_x, dtype=np.uint8)
        for chiplet in chiplets:
            x0 = float(chiplet["x"])
            y0 = float(chiplet["y"])
            x1 = x0 + float(chiplet["width"])
            y1 = y0 + float(chiplet["height"])

            inside = (
                (grid_x >= x0 - 1e-9)
                & (grid_x <= x1 + 1e-9)
                & (grid_y >= y0 - 1e-9)
                & (grid_y <= y1 + 1e-9)
            )
            occupancy |= inside.astype(np.uint8)
    edge_distance = np.full_like(grid_x, np.inf, dtype=np.float32)
    for chiplet in chiplets:
        x0 = float(chiplet["x"])
        y0 = float(chiplet["y"])
        x1 = x0 + float(chiplet["width"])
        y1 = y0 + float(chiplet["height"])
        edge_distance = np.minimum(
            edge_distance,
            _distance_to_rectangle_boundary(grid_x, grid_y, x0, y0, x1, y1),
        )

    if edge_band_mm is None:
        edge_band_mm = _default_edge_band_mm(grid_x, grid_y)
    edge_mask = (edge_distance <= float(edge_band_mm)).astype(np.uint8)

    coord_x_norm = (grid_x / max(outline_width, 1e-9)).astype(np.float32)
    coord_y_norm = (grid_y / max(outline_height, 1e-9)).astype(np.float32)

    return {
        "occupancy_mask": occupancy.astype(np.uint8),
        "edge_mask": edge_mask.astype(np.uint8),
        "coord_x_norm": coord_x_norm,
        "coord_y_norm": coord_y_norm,
    }


def augment_pointwise_sample(
    pointwise_csv: Path,
    sample_json: Path,
    output_csv: Path,
    *,
    edge_band_mm: float | None = None,
) -> dict[str, Any]:
    rows, fieldnames = _load_csv_rows(pointwise_csv)
    payload = _load_sample_json(sample_json)

    required_fields = {"grid_x", "grid_y", "chiplet_id"}
    missing_fields = required_fields.difference(fieldnames)
    if missing_fields:
        raise ValueError(f"{pointwise_csv} is missing columns: {sorted(missing_fields)}")

    grid_x = np.array([float(row["grid_x"]) for row in rows], dtype=np.float32)
    grid_y = np.array([float(row["grid_y"]) for row in rows], dtype=np.float32)
    chiplet_ids = np.array([str(row["chiplet_id"]) for row in rows], dtype=object)
    features = build_pointwise_feature_columns(
        grid_x,
        grid_y,
        payload,
        edge_band_mm=edge_band_mm,
        chiplet_ids=chiplet_ids,
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_fields = list(fieldnames)
    for name in features.keys():
        if name not in output_fields:
            output_fields.append(name)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields, extrasaction="raise")
        writer.writeheader()
        for index, row in enumerate(rows):
            output_row = {name: row.get(name, "") for name in fieldnames}
            missing_values = [name for name, value in output_row.items() if str(value).strip() == ""]
            if missing_values:
                raise ValueError(
                    f"{pointwise_csv} row {index} contains empty values for columns: {missing_values}"
                )
            output_row["occupancy_mask"] = int(features["occupancy_mask"][index])
            output_row["edge_mask"] = int(features["edge_mask"][index])
            output_row["coord_x_norm"] = f"{float(features['coord_x_norm'][index]):.6f}"
            output_row["coord_y_norm"] = f"{float(features['coord_y_norm'][index]):.6f}"
            writer.writerow(output_row)

    return {
        "input_pointwise": str(pointwise_csv),
        "input_json": str(sample_json),
        "output_csv": str(output_csv),
        "num_rows": len(rows),
        "edge_band_mm": float(edge_band_mm) if edge_band_mm is not None else None,
        "feature_columns": list(features.keys()),
    }
