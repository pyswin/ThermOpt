from pathlib import Path

import yaml

from thermopt.data.pointwise import load_pointwise_case
import thermopt.experiments.run_v0_sa as run_v0_module
from thermopt.experiments.run_v0_sa import run
from helpers import DummyThermalBackend


def write_pointwise_csv(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "grid_x,grid_y,chiplet_id,chiplet_power,temperature",
                "0,0,0,10,330",
                "1,0,0,10,331",
                "0,1,1,20,332",
                "1,1,1,20,333",
            ]
        ),
        encoding="utf-8",
    )


def test_load_pointwise_case_builds_case_and_layout(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample_000000.csv"
    write_pointwise_csv(csv_path)

    loaded = load_pointwise_case(
        csv_path,
        {
            "outline_width": 10.0,
            "outline_height": 10.0,
            "nearest_net_degree": 1,
        },
    )

    assert loaded.name == "sample_000000"
    assert len(loaded.case.chiplets) == 2
    assert len(loaded.case.nets) == 1
    assert len(loaded.layout.placements) == 2
    assert loaded.case.chiplets[0].power == 10
    assert loaded.case.chiplets[1].power == 20


def test_run_pointwise_config_writes_case_outputs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(run_v0_module, "build_thermal_backend", lambda *args, **kwargs: DummyThermalBackend())
    data_dir = tmp_path / "pointwise"
    data_dir.mkdir()
    write_pointwise_csv(data_dir / "sample_000000.csv")
    write_pointwise_csv(data_dir / "sample_000001.csv")

    config = {
        "seed": 3,
        "output_root": str(tmp_path / "outputs"),
        "experiment_name": "test_pointwise",
        "case": {
            "source": "pointwise",
            "data_dir": str(data_dir),
            "files": ["sample_000000.csv", "sample_000001.csv"],
            "outline_width": 20.0,
            "outline_height": 20.0,
            "nearest_net_degree": 1,
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

    assert (output_dir / "summary.json").exists()
    assert (output_dir / "sample_000000" / "metrics.csv").exists()
    assert (output_dir / "sample_000001" / "final_summary.png").exists()
