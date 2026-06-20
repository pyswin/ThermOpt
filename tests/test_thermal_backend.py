from pathlib import Path

from thermopt.layout.objects import Chiplet, FloorplanCase, Layout, Placement
from thermopt.optimizer.simulated_annealing import available_moves
from thermopt.thermal.backend import build_thermal_backend
from thermopt.thermal.hotspot import _render_hotspot_config, _write_hotspot_floorplans


def test_hotspot_backend_falls_back_when_binary_is_missing(tmp_path) -> None:
    case = FloorplanCase(
        chiplets=(Chiplet("A", 4.0, 4.0, 10.0), Chiplet("B", 4.0, 4.0, 20.0)),
        nets=(),
        outline_width=16.0,
        outline_height=12.0,
    )
    layout = Layout((Placement("A", 0.0, 0.0), Placement("B", 6.0, 0.0)))

    backend = build_thermal_backend(
        case,
        {
            "backend": "hotspot",
            "hotspot_binary": "",
            "hotspot_allow_fallback": True,
            "grid_size": [12, 10],
            "ambient": 25.0,
            "scale": 0.1,
            "sigma_factor": 1.0,
        },
        work_dir=tmp_path,
    )

    temperature = backend.simulate(case, layout)

    assert backend.name == "hotspot"
    assert backend.runtime_mode == "heuristic-fallback"
    assert temperature.shape == (10, 12)
    assert float(temperature.min()) >= 25.0


def test_sa_move_pool_can_disable_rotation() -> None:
    names, probs = available_moves(False)
    assert names == ["translate", "swap", "perturb"]
    assert abs(sum(probs) - 1.0) < 1e-9

    names_with_rotate, probs_with_rotate = available_moves(True)
    assert names_with_rotate == ["translate", "swap", "rotate", "perturb"]
    assert abs(sum(probs_with_rotate) - 1.0) < 1e-9


def test_hotspot_floorplan_respects_rotation(tmp_path: Path) -> None:
    case = FloorplanCase(
        chiplets=(Chiplet("A", 4.0, 6.0, 10.0),),
        nets=(),
        outline_width=20.0,
        outline_height=20.0,
    )
    layout = Layout((Placement("A", 2.0, 3.0, 90),))

    workspace = tmp_path / "hotspot"
    workspace.mkdir()
    _write_hotspot_floorplans(
        workspace,
        case,
        layout,
        {
            "grid_size": [8, 8],
            "ambient": 25.0,
            "scale": 0.1,
            "sigma_factor": 1.0,
        },
    )

    chip_layer = (workspace / "L4_ChipLayer.flp").read_text(encoding="utf-8").splitlines()
    chip_lines = [line for line in chip_layer if line and not line.startswith("#")]
    first_chip = next(line for line in chip_lines if line.startswith("Chiplet_0"))
    parts = first_chip.split("\t")
    assert float(parts[1]) == 6e-3
    assert float(parts[2]) == 4e-3


def test_hotspot_floorplan_expands_outline_and_uses_absolute_paths(tmp_path: Path) -> None:
    case = FloorplanCase(
        chiplets=(Chiplet("A", 4.0, 4.0, 10.0), Chiplet("B", 4.0, 4.0, 20.0)),
        nets=(),
        outline_width=20.0,
        outline_height=20.0,
    )
    layout = Layout((Placement("A", 0.0, 0.0), Placement("B", 17.0, 0.0)))

    workspace = tmp_path / "hotspot"
    workspace.mkdir()
    _write_hotspot_floorplans(
        workspace,
        case,
        layout,
        {
            "grid_size": [8, 8],
            "ambient": 25.0,
            "scale": 0.1,
            "sigma_factor": 1.0,
        },
    )

    lcf = (workspace / "layers.lcf").read_text(encoding="utf-8")
    assert str((workspace / "L0_Substrate.flp").resolve()) in lcf
    assert str((workspace / "L4_ChipLayer.flp").resolve()) in lcf

    base_lines = [line for line in (workspace / "L0_Substrate.flp").read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")]
    assert base_lines == ["Substrate\t0.021\t0.02\t0.0\t0.0"]

    chip_lines = [
        line
        for line in (workspace / "L4_ChipLayer.flp").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    assert not any(line.startswith("Edge_") for line in chip_lines)


def test_hotspot_config_uses_exact_rectangular_grid(tmp_path: Path) -> None:
    case = FloorplanCase(
        chiplets=(Chiplet("A", 4.0, 4.0, 10.0),),
        nets=(),
        outline_width=20.0,
        outline_height=12.0,
    )

    workspace = tmp_path / "hotspot"
    workspace.mkdir()
    config_path = _render_hotspot_config(
        {
            "grid_size": [12, 10],
            "ambient": 25.0,
        },
        case,
        workspace,
    )

    config_text = config_path.read_text(encoding="utf-8")
    grid_rows_line = next(line for line in config_text.splitlines() if "-grid_rows" in line)
    grid_cols_line = next(line for line in config_text.splitlines() if "-grid_cols" in line)
    assert grid_rows_line.strip().endswith("12")
    assert grid_cols_line.strip().endswith("10")
