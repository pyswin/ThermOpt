from pathlib import Path

import yaml

from thermopt.data.atplace import CASE_INTERPOSER_SIZE, load_atplace_case
from thermopt.experiments.run_v0_sa import run


def write_atplace_case(root: Path, name: str = "Case1") -> Path:
    case_dir = root / name
    case_dir.mkdir(parents=True)
    (case_dir / f"{name}.blocks").write_text(
        "\n".join(
            [
                "NumSoftRectangularBlocks : 0",
                "NumHardRectilinearBlocks : 2",
                "NumTerminals : 0",
                "",
                "CPU hardrectilinear 4 (0, 0) (0, 1000) (2000, 1000) (2000, 0)",
                "GPU hardrectilinear 4 (0, 0) (0, 3000) (3000, 3000) (3000, 0)",
            ]
        ),
        encoding="utf-8",
    )
    (case_dir / f"{name}.nets").write_text(
        "\n".join(
            [
                "NumNets : 1",
                "NumPins : 2",
                "",
                "NetDegree : 2",
                "CPU B : %0 %0",
                "GPU B : %0 %0",
            ]
        ),
        encoding="utf-8",
    )
    (case_dir / f"{name}.power").write_text("CPU 10\nGPU 30\n", encoding="utf-8")
    (case_dir / f"{name}.pl").write_text("CPU 0 0\nGPU 0 0\n", encoding="utf-8")
    return case_dir


def test_load_atplace_case_builds_case_from_bookshelf_files(tmp_path: Path) -> None:
    case_dir = write_atplace_case(tmp_path)

    loaded = load_atplace_case(
        case_dir,
        {
            "unit_scale": 0.001,
            "initial_layout": "random",
        },
        seed=11,
    )

    assert loaded.name == "Case1"
    assert loaded.case.outline_width == 42.0
    assert loaded.case.outline_height == 42.0
    assert len(loaded.case.chiplets) == 2
    assert len(loaded.case.nets) == 1
    assert loaded.case.chiplet_by_id["CPU"].width == 2.0
    assert loaded.case.chiplet_by_id["GPU"].power == 30
    assert len(loaded.layout.placements) == 2


def test_known_atplace_outline_sizes_match_paper() -> None:
    assert CASE_INTERPOSER_SIZE["Case1"] == (42000.0, 42000.0)
    assert CASE_INTERPOSER_SIZE["Case2"] == (55000.0, 52000.0)
    assert CASE_INTERPOSER_SIZE["Case3"] == (39000.0, 39000.0)
    assert CASE_INTERPOSER_SIZE["Case4"] == (57000.0, 59000.0)
    assert CASE_INTERPOSER_SIZE["Case5"] == (37000.0, 37000.0)


def test_run_atplace_config_writes_case_outputs(tmp_path: Path) -> None:
    data_dir = tmp_path / "cases"
    write_atplace_case(data_dir)
    config = {
        "seed": 3,
        "output_root": str(tmp_path / "outputs"),
        "experiment_name": "test_atplace",
        "case": {
            "source": "atplace",
            "data_dir": str(data_dir),
            "cases": ["Case1"],
            "unit_scale": 0.001,
            "initial_layout": "random",
        },
        "thermal": {"grid_size": [12, 12], "ambient": 25.0, "scale": 0.05, "sigma_factor": 1.0},
        "objective": {
            "alpha": 1.0,
            "beta": 1.0,
            "gamma": 20.0,
            "delta": 30.0,
            "thermal_mode": "topk",
            "topk_percent": 0.05,
            "temperature_limit": 85.0,
        },
        "optimizer": {
            "iterations": 2,
            "initial_anneal_temp": 1.0,
            "final_anneal_temp": 0.05,
            "move_scale": 2.0,
            "report_every": 1,
        },
        "experiments": [{"name": "wl_only", "objective": {"beta": 0.0}}],
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    output_dir = run(config_path)

    assert (output_dir / "metrics.csv").exists()
    assert (output_dir / "summary.json").exists()
