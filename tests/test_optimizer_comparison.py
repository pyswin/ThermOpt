from pathlib import Path

import yaml

from thermopt.experiments.run_optimizer_comparison import run


def test_optimizer_comparison_generates_rl_outputs(tmp_path: Path) -> None:
    config = {
        "seed": 40,
        "output_root": str(tmp_path),
        "experiment_name": "optimizer_compare_test",
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
        "thermal": {"grid_size": [24, 18], "ambient": 25.0, "scale": 1.0, "sigma_factor": 1.0},
        "objective": {
            "alpha": 1.0,
            "beta": 1.0,
            "gamma": 20.0,
            "delta": 30.0,
            "thermal_mode": "topk",
            "topk_percent": 0.05,
            "temperature_limit": 85.0,
        },
        "simulated_annealing": {
            "iterations": 8,
            "initial_anneal_temp": 1.0,
            "final_anneal_temp": 0.05,
            "move_scale": 4.0,
            "report_every": 2,
        },
        "genetic_algorithm": {
            "population_size": 6,
            "generations": 3,
            "elite_count": 2,
            "mutation_rate": 0.5,
            "tournament_size": 2,
            "move_scale": 4.0,
        },
        "reinforcement_learning": {
            "episodes": 3,
            "max_steps": 4,
            "rollout_steps": 4,
            "learning_rate": 0.02,
            "gamma": 0.95,
            "move_scale": 4.0,
            "verbose": False,
        },
    }
    config_path = tmp_path / "optimizer_comparison.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    output_dir = run(config_path)

    assert (output_dir / "final_layout_reinforcement_learning.png").exists()
    assert (output_dir / "final_temperature_reinforcement_learning.png").exists()
    assert (output_dir / "cost_curve_reinforcement_learning.png").exists()
    assert (output_dir / "optimizer_comparison_summary.png").exists()
    assert (output_dir / "metrics.csv").exists()
