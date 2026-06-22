from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from thermopt.data.inputs import CaseInput
from thermopt.layout.objects import Chiplet, FloorplanCase, Layout, Net, Placement


def _load_rows(path: Path) -> tuple[list[dict[str, object]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    return rows, fieldnames


def _grid_cell_area(rows: list[dict[str, object]]) -> float:
    xs = np.sort(np.unique([float(row["grid_x"]) for row in rows]))
    ys = np.sort(np.unique([float(row["grid_y"]) for row in rows]))
    dx = float(np.median(np.diff(xs))) if len(xs) > 1 else 1.0
    dy = float(np.median(np.diff(ys))) if len(ys) > 1 else 1.0
    return dx * dy


def _make_nets(placements: list[Placement], net_degree: int) -> tuple[Net, ...]:
    if len(placements) < 2:
        return ()

    centers = {
        placement.chiplet_id: np.array([placement.x, placement.y], dtype=float)
        for placement in placements
    }
    nets: list[Net] = []
    seen: set[tuple[str, str]] = set()
    degree = max(1, net_degree)

    for source in placements:
        distances = []
        for target in placements:
            if source.chiplet_id == target.chiplet_id:
                continue
            distance = float(np.linalg.norm(centers[source.chiplet_id] - centers[target.chiplet_id]))
            distances.append((distance, target.chiplet_id))

        for _, target_id in sorted(distances)[:degree]:
            endpoints = tuple(sorted((source.chiplet_id, target_id)))
            if endpoints in seen:
                continue
            seen.add(endpoints)
            nets.append(Net(id=f"N{len(nets)}", chiplets=endpoints))

    return tuple(nets)


def load_pointwise_case(path: Path, config: dict) -> CaseInput:
    rows, fieldnames = _load_rows(path)
    if not rows:
        raise ValueError(f"{path} is empty")
    required = {"grid_x", "grid_y", "chiplet_id", "chiplet_power"}
    missing = required.difference(fieldnames)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    grid_xs = np.array([float(row["grid_x"]) for row in rows], dtype=float)
    grid_ys = np.array([float(row["grid_y"]) for row in rows], dtype=float)
    outline_width = float(config.get("outline_width", float(grid_xs.max() - grid_xs.min())))
    outline_height = float(config.get("outline_height", float(grid_ys.max() - grid_ys.min())))
    scale_x = outline_width / max(float(grid_xs.max() - grid_xs.min()), 1e-9)
    scale_y = outline_height / max(float(grid_ys.max() - grid_ys.min()), 1e-9)
    cell_area = _grid_cell_area(rows) * scale_x * scale_y

    chiplets: list[Chiplet] = []
    placements: list[Placement] = []
    min_size = float(config.get("min_chiplet_size", 1.0))
    groups: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        chiplet_id = str(row["chiplet_id"])
        if chiplet_id == "background":
            continue
        groups.setdefault(chiplet_id, []).append(row)

    for chiplet_id in sorted(groups.keys()):
        group = groups[chiplet_id]
        name = f"C{chiplet_id}"
        area = max(float(len(group)) * cell_area, min_size * min_size)
        aspect = float(config.get("default_aspect_ratio", 1.0))
        width = max(min_size, float(np.sqrt(area * aspect)))
        height = max(min_size, float(area / width))
        cx = float(np.mean([float(row["grid_x"]) for row in group]) - grid_xs.min()) * scale_x
        cy = float(np.mean([float(row["grid_y"]) for row in group]) - grid_ys.min()) * scale_y
        x = min(max(0.0, cx - width * 0.5), max(0.0, outline_width - width))
        y = min(max(0.0, cy - height * 0.5), max(0.0, outline_height - height))

        chiplets.append(
            Chiplet(
                id=name,
                width=width,
                height=height,
                power=float(np.mean([float(row["chiplet_power"]) for row in group])),
            )
        )
        placements.append(Placement(chiplet_id=name, x=x, y=y))

    nets = _make_nets(placements, int(config.get("nearest_net_degree", 2)))
    case = FloorplanCase(
        chiplets=tuple(chiplets),
        nets=nets,
        outline_width=outline_width,
        outline_height=outline_height,
    )
    return CaseInput(path.stem, case, Layout(tuple(placements)), path)


def load_pointwise_cases(config: dict) -> list[CaseInput]:
    data_dir = Path(config.get("data_dir", "pointwise"))
    files = config.get("files")
    if files:
        paths = [data_dir / name for name in files]
    else:
        paths = sorted(data_dir.glob(str(config.get("pattern", "sample_*.csv"))))

    max_cases = config.get("max_cases")
    if max_cases is not None:
        paths = paths[: int(max_cases)]

    if not paths:
        raise ValueError(f"No pointwise CSV files found under {data_dir}")

    return [load_pointwise_case(path, config) for path in paths]
