from __future__ import annotations

import re
from pathlib import Path

from thermopt.data.case_generator import random_initial_layout
from thermopt.data.inputs import CaseInput
from thermopt.layout.objects import Chiplet, FloorplanCase, Layout, Net, Placement


CASE_INTERPOSER_SIZE = {
    "Case1": (42000.0, 42000.0),
    "Case2": (55000.0, 52000.0),
    "Case3": (39000.0, 39000.0),
    "Case4": (57000.0, 59000.0),
    "Case5": (37000.0, 37000.0),
    "Case6": (49000.0, 53000.0),
    "Case7": (30000.0, 25000.0),
    "Case8": (26000.0, 23000.0),
    "Case9": (59000.0, 61000.0),
    "Case10": (47000.0, 47000.0),
}


def _case_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)$", path.name)
    return (int(match.group(1)) if match else 10**9, path.name)


def _parse_blocks(path: Path) -> dict[str, tuple[float, float]]:
    chiplet_sizes: dict[str, tuple[float, float]] = {}
    pattern = re.compile(r"^(\S+)\s+hardrectilinear\s+\d+\s+(.+)$")
    point_pattern = re.compile(r"\(([-+0-9.eE]+),\s*([-+0-9.eE]+)\)")

    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        points = [(float(x), float(y)) for x, y in point_pattern.findall(match.group(2))]
        if not points:
            continue
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        chiplet_sizes[match.group(1)] = (max(xs) - min(xs), max(ys) - min(ys))

    if not chiplet_sizes:
        raise ValueError(f"No hardrectilinear chiplets found in {path}")
    return chiplet_sizes


def _parse_power(path: Path) -> dict[str, float]:
    powers: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        words = line.split()
        if len(words) >= 2:
            powers[words[0]] = float(words[1])
    return powers


def _parse_nets(path: Path, sizes: dict[str, tuple[float, float]], scale: float) -> tuple[Net, ...]:
    nets: list[Net] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        words = lines[i].split()
        if len(words) == 3 and words[0] == "NetDegree" and words[1] == ":":
            degree = int(words[2])
            pins: list[str] = []
            offsets: list[tuple[float, float]] = []
            for offset in range(1, degree + 1):
                if i + offset >= len(lines):
                    break
                pin_words = lines[i + offset].split()
                if pin_words:
                    chiplet_id = pin_words[0]
                    pins.append(chiplet_id)
                    pin_offset = (0.0, 0.0)
                    if len(pin_words) >= 5 and chiplet_id in sizes:
                        width, height = sizes[chiplet_id]
                        pin_offset = (
                            float(pin_words[3].lstrip("%")) * width * scale / 100.0,
                            float(pin_words[4].lstrip("%")) * height * scale / 100.0,
                        )
                    offsets.append(pin_offset)
            if len(pins) >= 2:
                nets.append(Net(id=f"N{len(nets)}", chiplets=tuple(pins), pin_offsets=tuple(offsets)))
            i += degree + 1
        else:
            i += 1
    return tuple(nets)


def _parse_pl_layout(path: Path, case: FloorplanCase, scale: float) -> Layout:
    chiplets = case.chiplet_by_id
    placements: list[Placement] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        words = line.split()
        if len(words) < 3 or words[0] not in chiplets:
            continue
        chiplet = chiplets[words[0]]
        cx = float(words[1]) * scale
        cy = float(words[2]) * scale
        placements.append(Placement(chiplet.id, cx - chiplet.width * 0.5, cy - chiplet.height * 0.5))
    return Layout(tuple(placements))


def load_atplace_case(case_dir: Path, config: dict, seed: int) -> CaseInput:
    case_name = case_dir.name
    scale = float(config.get("unit_scale", 0.001))
    blocks_path = case_dir / f"{case_name}.blocks"
    nets_path = case_dir / f"{case_name}.nets"
    power_path = case_dir / f"{case_name}.power"
    pl_path = case_dir / f"{case_name}.pl"

    sizes = _parse_blocks(blocks_path)
    powers = _parse_power(power_path)
    chiplets = tuple(
        Chiplet(id=name, width=width * scale, height=height * scale, power=powers.get(name, 1.0))
        for name, (width, height) in sizes.items()
    )
    raw_outline = CASE_INTERPOSER_SIZE.get(case_name)
    if raw_outline is None:
        max_width = max(width for width, _ in sizes.values())
        max_height = max(height for _, height in sizes.values())
        raw_outline = (max_width * 3.0, max_height * 3.0)

    case = FloorplanCase(
        chiplets=chiplets,
        nets=_parse_nets(nets_path, sizes, scale),
        outline_width=raw_outline[0] * scale,
        outline_height=raw_outline[1] * scale,
    )

    layout_mode = str(config.get("initial_layout", "random")).lower()
    if layout_mode == "pl":
        layout = _parse_pl_layout(pl_path, case, scale)
    elif layout_mode == "random":
        layout = random_initial_layout(case, seed)
    else:
        raise ValueError(f"Unknown ATPlace initial_layout: {layout_mode}")

    return CaseInput(case_name, case, layout, case_dir)


def load_atplace_cases(config: dict, seed: int) -> list[CaseInput]:
    data_dir = Path(config.get("data_dir", "external/ATPlace_pub/cases"))
    names = config.get("cases")
    if names:
        case_dirs = [data_dir / name for name in names]
    else:
        case_dirs = sorted((path for path in data_dir.iterdir() if path.is_dir()), key=_case_sort_key)

    max_cases = config.get("max_cases")
    if max_cases is not None:
        case_dirs = case_dirs[: int(max_cases)]

    if not case_dirs:
        raise ValueError(f"No ATPlace case directories found under {data_dir}")

    return [load_atplace_case(path, config, seed + index) for index, path in enumerate(case_dirs)]
