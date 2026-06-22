from __future__ import annotations

import hashlib
import platform
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from thermopt.layout.geometry import bounds
from thermopt.layout.objects import FloorplanCase, Layout

_FLOORPLAN_HEADER = [
    "# Line Format: <unit-name>\\t<width>\\t<height>\\t<left-x>\\t<bottom-y>\\t[<specific-heat>]\\t[<resistivity>]\n",
    "# all dimensions are in meters\n",
    "# comment lines begin with a '#'\n",
    "# comments and empty lines are ignored\n\n",
]


@dataclass(frozen=True)
class _FlpItem:
    name: str
    width: float
    height: float
    x: float
    y: float
    tail: str = ""


def _mm_to_m(value: float) -> float:
    return float(value) * 1e-3


def _layout_signature(case: FloorplanCase, layout: Layout) -> str:
    by_id = layout.by_id
    payload = []
    for chiplet_id in case.chiplet_ids:
        placement = by_id.get(chiplet_id)
        if placement is None:
            raise ValueError(f"layout missing chiplet {chiplet_id}")
        payload.append(
            (
                chiplet_id,
                round(float(placement.x), 6),
                round(float(placement.y), 6),
                int(placement.rotation) % 360,
            )
        )
    digest = hashlib.sha1(repr(payload).encode("utf-8")).hexdigest()[:16]
    return digest


def _grid_size(config: dict) -> tuple[int, int]:
    if "grid_size" in config:
        raw = config["grid_size"]
        if isinstance(raw, (list, tuple)) and len(raw) == 2:
            return int(raw[0]), int(raw[1])
    if "num_grid_x" in config and "num_grid_y" in config:
        return int(config["num_grid_x"]), int(config["num_grid_y"])
    return 100, 80


def _next_power_of_two(value: int) -> int:
    value = max(1, int(value))
    return 1 << (value - 1).bit_length()


def _hotspot_grid_size(config: dict) -> tuple[int, int]:
    rows, cols = _grid_size(config)
    return _next_power_of_two(rows), _next_power_of_two(cols)


def _hotspot_template_path() -> Path:
    repo_candidate = _repo_root() / "external" / "ATPlace_pub" / "thermal" / "hotspot.config"
    if repo_candidate.is_file():
        return repo_candidate
    return Path(__file__).with_name("hotspot.config")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _platform_hotspot_names() -> list[str]:
    system = platform.system().lower()
    machine = platform.machine().lower()
    names: list[str] = []
    if system == "darwin":
        names.append(f"hotspot-darwin-{machine}")
        if machine in {"arm64", "aarch64"}:
            names.append("hotspot-darwin-arm64")
        elif machine in {"x86_64", "amd64"}:
            names.append("hotspot-darwin-x86_64")
    elif system == "linux":
        names.append(f"hotspot-linux-{machine}")
        if machine in {"x86_64", "amd64"}:
            names.append("hotspot")
            names.append("hotspot-linux-x86_64")
        elif machine in {"arm64", "aarch64"}:
            names.append("hotspot-linux-arm64")
    return list(dict.fromkeys(names))


def _is_compatible_binary(path: Path) -> bool:
    if not path.is_file():
        return False
    system = platform.system().lower()
    try:
        header = path.read_bytes()[:4]
    except OSError:
        return False
    if system == "darwin":
        return header in {b"\xfe\xed\xfa\xce", b"\xfe\xed\xfa\xcf", b"\xca\xfe\xba\xbe", b"\xcf\xfa\xed\xfe"}
    if system == "linux":
        return header == b"\x7fELF"
    return True


def _replace_config_value(lines: list[str], key: str, value: str) -> list[str]:
    pattern = re.compile(rf"^(\s*{re.escape(key)}\s+).*$")
    replaced: list[str] = []
    done = False
    for line in lines:
        if pattern.match(line):
            replaced.append(pattern.sub(lambda match: f"{match.group(1)}{value}", line))
            done = True
        else:
            replaced.append(line)
    if not done:
        replaced.append(f"\t\t{key}\t\t{value}\n")
    return replaced


def _render_hotspot_config(config: dict, case: FloorplanCase, workspace: Path) -> Path:
    template_path = _hotspot_template_path()
    if not template_path.exists():
        raise FileNotFoundError(f"missing HotSpot template: {template_path}")

    lines = template_path.read_text(encoding="utf-8").splitlines(keepends=True)
    grid_x, grid_y = _hotspot_grid_size(config)
    ambient_k = float(config.get("ambient", 25.0)) + 273.15
    thermal_threshold = float(config.get("thermal_threshold", ambient_k + 80.0))
    spreader_size = float(config.get("spreader_size", (case.outline_width + case.outline_height) / 1000.0))
    sink_size = float(config.get("sink_size", spreader_size * 2.0))
    r_convec = float(config.get("r_convec", 0.1 * 0.06 * 0.06 / max(sink_size, 1e-9) / max(sink_size, 1e-9)))

    lines = _replace_config_value(lines, "-grid_rows", str(grid_x))
    lines = _replace_config_value(lines, "-grid_cols", str(grid_y))
    lines = _replace_config_value(lines, "-ambient", f"{ambient_k:.6f}")
    lines = _replace_config_value(lines, "-init_temp", f"{ambient_k:.6f}")
    lines = _replace_config_value(lines, "-thermal_threshold", f"{thermal_threshold:.6f}")
    lines = _replace_config_value(lines, "-s_spreader", f"{spreader_size:.6f}")
    lines = _replace_config_value(lines, "-s_sink", f"{sink_size:.6f}")
    lines = _replace_config_value(lines, "-r_convec", f"{r_convec:.6f}")

    config_path = workspace / "new_hotspot.config"
    config_path.write_text("".join(lines), encoding="utf-8")
    return config_path


def _material_tails() -> dict[str, str]:
    resistivity_cu, spec_heat_cu = 0.0025, 3494400.0
    resistivity_uf, spec_heat_uf = 0.625, 2320000.0
    resistivity_si, spec_heat_si = 0.01, 1750000.0
    c4_diameter, c4_edge = 0.000250, 0.000600
    tsv_diameter, tsv_edge = 0.000010, 0.000050
    ubump_diameter, ubump_edge = 0.000025, 0.000045

    aratio_c4 = (c4_edge / c4_diameter) * (c4_edge / c4_diameter) - 1.0
    aratio_tsv = (tsv_edge / tsv_diameter) * (tsv_edge / tsv_diameter) - 1.0
    aratio_ubump = (ubump_edge / ubump_diameter) * (ubump_edge / ubump_diameter) - 1.0

    resistivity_c4 = (1.0 + aratio_c4) * resistivity_cu * resistivity_uf / (
        resistivity_uf + aratio_c4 * resistivity_cu
    )
    resistivity_tsv = (1.0 + aratio_tsv) * resistivity_cu * resistivity_si / (
        resistivity_si + aratio_tsv * resistivity_cu
    )
    resistivity_ubump = (1.0 + aratio_ubump) * resistivity_cu * resistivity_uf / (
        resistivity_uf + aratio_ubump * resistivity_cu
    )

    spec_heat_c4 = (spec_heat_cu + aratio_c4 * spec_heat_uf) / (1.0 + aratio_c4)
    spec_heat_tsv = (spec_heat_cu + aratio_tsv * spec_heat_si) / (1.0 + aratio_tsv)
    spec_heat_ubump = (spec_heat_cu + aratio_ubump * spec_heat_uf) / (1.0 + aratio_ubump)

    return {
        "underfill": f"\t{spec_heat_uf:.2E}\t{resistivity_uf}\n",
        "silicon": f"\t{spec_heat_si:.2E}\t{resistivity_si}\n",
        "ubump": f"\t{spec_heat_ubump}\t{resistivity_ubump}\n",
        "c4": f"\t{spec_heat_c4}\t{resistivity_c4}\n",
        "tsv": f"\t{spec_heat_tsv}\t{resistivity_tsv}\n",
    }


def _write_floorplan_file(
    path: Path,
    name: str,
    width_m: float,
    height_m: float,
    background_tail: str = "",
    chiplets: list[_FlpItem] | None = None,
    include_background: bool = True,
) -> None:
    body: list[str] = [f"# Floorplan for {name}\n", *_FLOORPLAN_HEADER]
    if include_background:
        body.append(f"{name}\t{width_m}\t{height_m}\t0.0\t0.0{background_tail}\n")
    if chiplets:
        for item in chiplets:
            body.append(f"{item.name}\t{item.width}\t{item.height}\t{item.x}\t{item.y}{item.tail}\n")
    path.write_text("".join(body), encoding="utf-8")


def _write_lcf(path: Path, workspace: Path) -> None:
    workspace = workspace.resolve()
    lines = [
        "# File Format:\n",
        "#<Layer Number>\n",
        "#<Lateral heat flow Y/N?>\n",
        "#<Power Dissipation Y/N?>\n",
        "#<Specific heat capacity in J/(m^3K)>\n",
        "#<Resistivity in (m-K)/W>\n",
        "#<Thickness in m>\n",
        "#<floorplan file>\n",
        "\n# Layer 0: substrate\n0\nY\nN\n1.06E+06\n3.33\n0.0002\n",
        f"{(workspace / 'L0_Substrate.flp').resolve()}\n",
        "\n# Layer 1: Epoxy SiO2 underfill with C4 copper pillar\n1\nY\nN\n2.32E+06\n0.625\n0.00007\n",
        f"{(workspace / 'L1_C4Layer.flp').resolve()}\n",
        "\n# Layer 2: silicon interposer\n2\nY\nN\n1.75E+06\n0.01\n0.00011\n",
        f"{(workspace / 'L2_Interposer.flp').resolve()}\n",
        "\n# Layer 3: Underfill with ubump\n3\nY\nN\n2.32E+06\n0.625\n1.00E-05\n",
        f"{(workspace / 'L3_UbumpLayer.flp').resolve()}\n",
        "\n# Layer 4: Chip layer\n4\nY\nY\n1.75E+06\n0.01\n0.00015\n",
        f"{(workspace / 'L4_ChipLayer.flp').resolve()}\n",
        "\n# Layer 5: TIM\n5\nY\nN\n4.00E+06\n0.25\n2.00E-05\n",
        f"{(workspace / 'L5_TIM.flp').resolve()}\n",
    ]
    path.write_text("".join(lines), encoding="utf-8")


def _write_ptrace(path: Path, unit_names: list[str], power_by_unit: dict[str, float]) -> None:
    header = "\t".join(unit_names) + "\n"
    powers = []
    for unit_name in unit_names:
        power = power_by_unit.get(unit_name, 0.0)
        powers.append(f"{power:.6f}")
    path.write_text(header + "\t".join(powers) + "\n", encoding="utf-8")


def _layout_extent_m(case: FloorplanCase, layout: Layout) -> tuple[float, float]:
    max_x = 0.0
    max_y = 0.0
    for chiplet_id in case.chiplet_ids:
        placement = layout.by_id[chiplet_id]
        _, _, x1, y1 = bounds(case, placement)
        max_x = max(max_x, _mm_to_m(x1))
        max_y = max(max_y, _mm_to_m(y1))
    return max_x, max_y


def _hotspot_box(case: FloorplanCase, layout: Layout, config: dict) -> tuple[float, float, float]:
    base_width = _mm_to_m(case.outline_width)
    base_height = _mm_to_m(case.outline_height)
    layout_width, layout_height = _layout_extent_m(case, layout)
    width_m = max(base_width, layout_width)
    height_m = max(base_height, layout_height)
    requested_edge = max(0.0, float(config.get("edge_thickness", 0.00005)))
    fits_requested_edge = True
    if width_m <= base_width + 1e-12 and height_m <= base_height + 1e-12:
        inner_x0 = requested_edge
        inner_y0 = requested_edge
        inner_x1 = base_width - requested_edge
        inner_y1 = base_height - requested_edge
        for chiplet_id in case.chiplet_ids:
            placement = layout.by_id[chiplet_id]
            x0, y0, x1, y1 = bounds(case, placement)
            if (
                _mm_to_m(x0) < inner_x0 - 1e-12
                or _mm_to_m(y0) < inner_y0 - 1e-12
                or _mm_to_m(x1) > inner_x1 + 1e-12
                or _mm_to_m(y1) > inner_y1 + 1e-12
            ):
                fits_requested_edge = False
                break
    else:
        fits_requested_edge = False
    edge = requested_edge if fits_requested_edge else 0.0
    return width_m, height_m, edge


def _fill_space(
    width_st: float,
    width_ed: float,
    height_st: float,
    height_ed: float,
    occupied: list[_FlpItem],
    ws_tail: str = "",
) -> list[_FlpItem]:
    eps = 1e-5
    ws: list[_FlpItem] = []
    ws_n = 0

    def _covers(item: _FlpItem, x0: float, x1: float, y0: float, y1: float) -> bool:
        return (
            x0 >= item.x - eps
            and y0 >= item.y - eps
            and x1 <= item.x + item.width + eps
            and y1 <= item.y + item.height + eps
        )

    xs = {float(width_st), float(width_ed)}
    ys = {float(height_st), float(height_ed)}
    for item in occupied:
        xs.add(max(float(width_st), float(item.x)))
        xs.add(min(float(width_ed), float(item.x + item.width)))
        ys.add(max(float(height_st), float(item.y)))
        ys.add(min(float(height_ed), float(item.y + item.height)))
    x_edges = sorted(xs)
    y_edges = sorted(ys)

    for x0, x1 in zip(x_edges, x_edges[1:]):
        if x1 - x0 < eps:
            continue
        for y0, y1 in zip(y_edges, y_edges[1:]):
            if y1 - y0 < eps:
                continue
            if any(_covers(item, x0, x1, y0, y1) for item in occupied):
                continue
            ws.append(_FlpItem(f"WS_{ws_n}", x1 - x0, y1 - y0, x0, y0, ws_tail))
            ws_n += 1

    return ws


def _write_hotspot_floorplans(workspace: Path, case: FloorplanCase, layout: Layout, config: dict) -> Path:
    chiplets = case.chiplet_by_id
    width_m, height_m, edge = _hotspot_box(case, layout, config)
    if edge > 0.0:
        x0 = edge
        x1 = max(edge, width_m - edge)
        y0 = edge
        y1 = max(edge, height_m - edge)
        if x1 < x0 or y1 < y0:
            raise ValueError("interposer outline is too small for the configured edge thickness")
    else:
        x0 = 0.0
        x1 = width_m
        y0 = 0.0
        y1 = height_m

    tails = _material_tails()
    silicon_tail = tails["silicon"]
    underfill_tail = tails["underfill"]
    ubump_tail = tails["ubump"]
    mat_c4_tail = tails["c4"]
    mat_tsv_tail = tails["tsv"]

    chiplet_items_l3: list[_FlpItem] = []
    chiplet_items_l4: list[_FlpItem] = []
    sim_items: list[_FlpItem] = []
    power_by_unit: dict[str, float] = {}
    for chiplet_id in case.chiplet_ids:
        placement = layout.by_id[chiplet_id]
        chiplet = chiplets[chiplet_id]
        unit_name = f"Chiplet_{len(chiplet_items_l4)}"
        power_by_unit[unit_name] = float(chiplet.power)
        x_left, y_bottom, _, _ = bounds(case, placement)
        x_left_m = _mm_to_m(x_left)
        y_bottom_m = _mm_to_m(y_bottom)
        width_mm, height_mm = placement.rotated_size(chiplet)
        width_chiplet_m = _mm_to_m(width_mm)
        height_chiplet_m = _mm_to_m(height_mm)
        chiplet_items_l3.append(_FlpItem(unit_name, width_chiplet_m, height_chiplet_m, x_left_m, y_bottom_m, ubump_tail))
        chiplet_items_l4.append(_FlpItem(unit_name, width_chiplet_m, height_chiplet_m, x_left_m, y_bottom_m, silicon_tail))
        sim_items.append(_FlpItem(f"Unit_{len(sim_items)}", width_chiplet_m, height_chiplet_m, x_left_m, y_bottom_m))

    ws_items_l3 = _fill_space(x0, x1, y0, y1, chiplet_items_l3, underfill_tail)
    ws_items_l4 = _fill_space(x0, x1, y0, y1, chiplet_items_l4, underfill_tail)

    edge_units: list[_FlpItem] = []
    if edge > 0.0:
        edge_units = [
            _FlpItem("Edge_0", max(0.0, width_m - 2.0 * edge), edge, edge, 0.0, ubump_tail),
            _FlpItem("Edge_1", max(0.0, width_m - 2.0 * edge), edge, edge, max(0.0, height_m - edge), ubump_tail),
            _FlpItem("Edge_2", edge, height_m, 0.0, 0.0, ubump_tail),
            _FlpItem("Edge_3", edge, height_m, max(0.0, width_m - edge), 0.0, ubump_tail),
        ]

    ordered_units = [item.name for item in edge_units + chiplet_items_l4 + ws_items_l4]
    power_by_unit.update({item.name: 0.0 for item in edge_units})
    power_by_unit.update({item.name: 0.0 for item in ws_items_l4})

    _write_floorplan_file(workspace / "L0_Substrate.flp", "Substrate", width_m, height_m)
    _write_floorplan_file(workspace / "L1_C4Layer.flp", "C4Layer", width_m, height_m, background_tail=mat_c4_tail)
    _write_floorplan_file(workspace / "L2_Interposer.flp", "Interposer", width_m, height_m, background_tail=mat_tsv_tail)
    _write_floorplan_file(
        workspace / "L3_UbumpLayer.flp",
        "Chip Layer",
        width_m,
        height_m,
        chiplets=edge_units + chiplet_items_l3 + ws_items_l3,
        include_background=False,
    )
    _write_floorplan_file(
        workspace / "L4_ChipLayer.flp",
        "Chip Layer",
        width_m,
        height_m,
        chiplets=edge_units + chiplet_items_l4 + ws_items_l4,
        include_background=False,
    )
    _write_floorplan_file(workspace / "L5_TIM.flp", "TIM", width_m, height_m)

    _write_lcf(workspace / "layers.lcf", workspace)
    _write_ptrace(workspace / "sample.ptrace", ordered_units, power_by_unit)
    _write_floorplan_file(workspace / "sim.flp", "SIM", width_m, height_m, chiplets=sim_items, include_background=False)
    return workspace / "L4_ChipLayer.flp"


def _parse_grid_values(path: Path, expected: int, preferred_layer: int = 4) -> np.ndarray:
    layers: dict[int, list[float]] = {}
    flat_values: list[float] = []
    current_layer: int | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        layer_match = re.fullmatch(r"Layer\s+(\d+):", line)
        if layer_match:
            current_layer = int(layer_match.group(1))
            layers.setdefault(current_layer, [])
            continue
        parts = line.split()
        try:
            value = float(parts[1] if len(parts) >= 2 else parts[0])
        except (IndexError, ValueError):
            continue
        if current_layer is None:
            flat_values.append(value)
        else:
            layers.setdefault(current_layer, []).append(value)

    if layers:
        if preferred_layer in layers and len(layers[preferred_layer]) == expected:
            return np.array(layers[preferred_layer], dtype=float)
        complete_layers = [values for _, values in sorted(layers.items()) if len(values) == expected]
        if complete_layers:
            return np.array(complete_layers[-1], dtype=float)
        layer_sizes = {layer: len(values) for layer, values in sorted(layers.items())}
        raise ValueError(f"unexpected HotSpot grid layer sizes in {path}: {layer_sizes}, expected {expected}")

    return np.array(flat_values, dtype=float)


def _parse_grid_steady(path: Path, grid_size: tuple[int, int]) -> np.ndarray:
    rows, cols = grid_size
    expected = rows * cols
    values = _parse_grid_values(path, expected)
    if values.size != expected:
        raise ValueError(f"unexpected HotSpot grid size in {path}: got {values.size}, expected {expected}")
    temp = values.reshape(rows, cols)
    temp = np.transpose(temp, (1, 0))[:, ::-1]
    return temp - 273.15


def _resample_grid(grid: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    target_rows, target_cols = target_size
    if grid.shape == (target_rows, target_cols):
        return grid

    src_rows, src_cols = grid.shape
    src_x = np.linspace(0.0, 1.0, src_cols)
    src_y = np.linspace(0.0, 1.0, src_rows)
    tgt_x = np.linspace(0.0, 1.0, target_cols)
    tgt_y = np.linspace(0.0, 1.0, target_rows)

    row_interp = np.array([np.interp(tgt_x, src_x, row) for row in grid], dtype=float)
    col_interp = np.array([np.interp(tgt_y, src_y, row_interp[:, col]) for col in range(row_interp.shape[1])], dtype=float)
    return col_interp.T


@dataclass
class HotSpotBackend:
    case: FloorplanCase
    config: dict
    work_dir: Path | None = None
    name: str = "hotspot"
    _cache: dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)
    _workspace_root: Path = field(init=False, repr=False)
    _binary_path: Path | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self._workspace_root = (
            Path(self.work_dir).expanduser().resolve()
            if self.work_dir is not None
            else Path(tempfile.mkdtemp(prefix="thermopt-hotspot-")).resolve()
        )
        self._workspace_root.mkdir(parents=True, exist_ok=True)
        self._binary_path = self._resolve_binary()
        if self._binary_path is None and bool(self.config.get("hotspot_required", True)):
            raise FileNotFoundError(
                "HotSpot binary not found. Set thermal.hotspot_binary or explicitly disable hotspot_required."
            )

    @property
    def runtime_mode(self) -> str:
        return "hotspot"

    def _resolve_binary(self) -> Path | None:
        binary = str(self.config.get("hotspot_binary", "hotspot")).strip()
        if not binary:
            return None
        repo_root = _repo_root()
        vendor_root = repo_root / "external" / "ATPlace_pub" / "thermal"
        candidates: list[Path] = []

        for name in _platform_hotspot_names():
            candidates.append(vendor_root / name)

        candidate = Path(binary).expanduser()
        candidates.append(candidate)
        if not candidate.is_absolute():
            candidates.append((repo_root / candidate).resolve())

        candidates.append(vendor_root / "hotspot")

        found = shutil.which(binary)
        if found:
            candidates.append(Path(found).resolve())

        for name in _platform_hotspot_names():
            candidates.append(repo_root / "third_party" / "HotSpot" / name)
        candidates.append(repo_root / "third_party" / "HotSpot" / "hotspot")

        for path in dict.fromkeys(candidates):
            resolved = path.expanduser().resolve()
            if _is_compatible_binary(resolved):
                return resolved
        return None

    def _layout_key(self, layout: Layout) -> str:
        return _layout_signature(self.case, layout)

    def _workspace_for(self, layout_key: str) -> Path:
        path = (self._workspace_root / layout_key).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def simulate(self, case: FloorplanCase, layout: Layout) -> np.ndarray:
        if case != self.case:
            raise ValueError("HotSpotBackend is bound to a different case")

        key = self._layout_key(layout)
        if key in self._cache:
            return np.array(self._cache[key], copy=True)

        if self._binary_path is None:
            raise FileNotFoundError("HotSpot binary not found. Set thermal.hotspot_binary to a compatible executable.")

        workspace = self._workspace_for(key)
        try:
            chip_layer = _write_hotspot_floorplans(workspace, case, layout, self.config)
            config_path = _render_hotspot_config(self.config, case, workspace)
            steady_file = workspace / "sample.steady"
            grid_file = workspace / "sample.grid.steady"
            cmd = [
                str(self._binary_path),
                "-c",
                str(config_path),
                "-f",
                str(chip_layer),
                "-p",
                str(workspace / "sample.ptrace"),
                "-steady_file",
                str(steady_file),
                "-grid_steady_file",
                str(grid_file),
                "-model_type",
                "grid",
                "-detailed_3D",
                "on",
                "-grid_layer_file",
                str(workspace / "layers.lcf"),
            ]
            timeout = self.config.get("hotspot_timeout_sec")
            subprocess.run(
                cmd,
                cwd=workspace,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=float(timeout) if timeout is not None else None,
            )
            if not grid_file.exists():
                raise FileNotFoundError(f"HotSpot did not produce grid output: {grid_file}")
            temp = _parse_grid_steady(grid_file, _hotspot_grid_size(self.config))
            temp = _resample_grid(temp, _grid_size(self.config))
        except Exception:
            raise

        self._cache[key] = np.array(temp, copy=True)
        return np.array(temp, copy=True)
