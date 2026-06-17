from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from thermopt.data.inputs import CaseInput
from thermopt.layout.objects import Chiplet, FloorplanCase, Layout, Net, Placement


def _grid_cell_area(df: pd.DataFrame) -> float:
    xs = np.sort(df["grid_x"].unique())
    ys = np.sort(df["grid_y"].unique())
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
    df = pd.read_csv(path)
    required = {"grid_x", "grid_y", "chiplet_id", "chiplet_power"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    outline_width = float(config.get("outline_width", df["grid_x"].max() - df["grid_x"].min()))
    outline_height = float(config.get("outline_height", df["grid_y"].max() - df["grid_y"].min()))
    scale_x = outline_width / max(float(df["grid_x"].max() - df["grid_x"].min()), 1e-9)
    scale_y = outline_height / max(float(df["grid_y"].max() - df["grid_y"].min()), 1e-9)
    cell_area = _grid_cell_area(df) * scale_x * scale_y

    chiplets: list[Chiplet] = []
    placements: list[Placement] = []
    min_size = float(config.get("min_chiplet_size", 1.0))

    for chiplet_id, group in df.groupby("chiplet_id", sort=True):
        name = f"C{chiplet_id}"
        area = max(float(len(group)) * cell_area, min_size * min_size)
        aspect = float(config.get("default_aspect_ratio", 1.0))
        width = max(min_size, float(np.sqrt(area * aspect)))
        height = max(min_size, float(area / width))
        cx = float((group["grid_x"].mean() - df["grid_x"].min()) * scale_x)
        cy = float((group["grid_y"].mean() - df["grid_y"].min()) * scale_y)
        x = min(max(0.0, cx - width * 0.5), max(0.0, outline_width - width))
        y = min(max(0.0, cy - height * 0.5), max(0.0, outline_height - height))

        chiplets.append(
            Chiplet(
                id=name,
                width=width,
                height=height,
                power=float(group["chiplet_power"].mean()),
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
