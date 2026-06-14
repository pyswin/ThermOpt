from pathlib import Path

import yaml

from thermopt.experiments.run_v0_sa import run


def test_run_generates_final_summary(tmp_path: Path) -> None:
    config = {
        "seed": 3,
        "output_root": str(tmp_path),
        "experiment_name": "test_v0",
        "case": {
            "num_chiplets": 5,
            "outline_width": 50.0,
            "outline_height": 40.0,
            "min_size": 5.0,
            "max_size": 10.0,
            "min_power": 1.0,
            "max_power": 5.0,
            "hot_chiplet_fraction": 0.2,
            "hot_power_multiplier": 2.0,
            "num_nets": 4,
            "net_min_degree": 2,
            "net_max_degree": 3,
        },
        "thermal": {
            "grid_size": [30, 20],
            "ambient": 25.0,
            "scale": 1.0,
            "sigma_factor": 1.0,
        },
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
            "iterations": 8,
            "initial_anneal_temp": 1.0,
            "final_anneal_temp": 0.05,
            "move_scale": 5.0,
            "report_every": 2,
        },
        "experiments": [
            {"name": "wl_only", "objective": {"beta": 0.0}},
            {"name": "wl_tmax", "objective": {"beta": 1.0, "thermal_mode": "tmax"}},
            {"name": "wl_topk", "objective": {"beta": 1.0, "thermal_mode": "topk"}},
        ],
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    output_dir = run(config_path)

    assert (output_dir / "final_summary.png").exists()
    assert (output_dir / "metrics.csv").exists()
    assert (output_dir / "summary.json").exists()
