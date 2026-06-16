from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from thermopt.data.case_generator import generate_random_case, random_initial_layout
from thermopt.layout.visualization import save_cost_curve, save_final_summary, save_layout_figure, save_temperature_figure
from thermopt.objective.cost import Objective
from thermopt.optimizer import genetic_algorithm, rl_policy, simulated_annealing
from thermopt.thermal.heuristic import simulate_temperature


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_output_dir(config: dict) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.get("output_root", "outputs")) / f"{stamp}_{config.get('experiment_name', 'optimizer_comparison')}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def run(config_path: Path) -> Path:
    config = load_config(config_path)
    output_dir = make_output_dir(config)
    shutil.copy2(config_path, output_dir / "config.yaml")

    seed = int(config.get("seed", 0))
    case = generate_random_case(config["case"], seed)
    initial_layout = random_initial_layout(case, seed + 1)
    objective = Objective(case, config["thermal"], config["objective"], initial_layout)
    initial_temperature = simulate_temperature(case, initial_layout, config["thermal"])
    save_layout_figure(case, initial_layout, output_dir / "initial_layout.png", "Initial layout")
    save_temperature_figure(initial_temperature, output_dir / "initial_temperature.png", "Initial temperature")

    runs = [
        ("simulated_annealing", simulated_annealing.optimize, config["simulated_annealing"], seed + 100),
        ("genetic_algorithm", genetic_algorithm.optimize, config["genetic_algorithm"], seed + 200),
        ("reinforcement_learning", rl_policy.optimize, config["reinforcement_learning"], seed + 300),
    ]

    rows: list[dict] = []
    final_results: list[dict] = []
    summary = {
        "seed": seed,
        "config": str(config_path),
        "output_dir": str(output_dir),
        "optimizers": {},
    }

    for name, optimizer, optimizer_config, optimizer_seed in runs:
        print(f"[compare] start optimizer={name}")
        started = time.perf_counter()
        result = optimizer(case, initial_layout, objective, optimizer_config, optimizer_seed)
        runtime = time.perf_counter() - started
        final_temperature = simulate_temperature(case, result.best_layout, config["thermal"])

        save_layout_figure(case, result.best_layout, output_dir / f"final_layout_{name}.png", f"Final layout: {name}")
        save_temperature_figure(final_temperature, output_dir / f"final_temperature_{name}.png", f"Final temperature: {name}")
        save_cost_curve(result.best_curve, output_dir / f"cost_curve_{name}.png", f"Cost curve: {name}")

        row = {
            "optimizer": name,
            "runtime_sec": runtime,
            **result.best_cost.metrics,
        }
        if hasattr(result, "accepted_ratio"):
            row["accepted_ratio"] = result.accepted_ratio
        if hasattr(result, "population_size"):
            row["population_size"] = result.population_size
        if hasattr(result, "training_episodes"):
            row["training_episodes"] = result.training_episodes
            row["rollout_steps"] = result.rollout_steps
            row["mean_episode_return"] = float(sum(result.episode_returns) / max(1, len(result.episode_returns)))

        print(
            f"[compare] done optimizer={name} runtime={runtime:.2f}s "
            f"cost={result.best_cost.total:.4f} wl={result.best_cost.metrics['wirelength']:.2f} "
            f"tmax={result.best_cost.metrics['tmax']:.2f} top5={result.best_cost.metrics['top5']:.2f}"
        )
        rows.append(row)
        summary["optimizers"][name] = row
        final_results.append(
            {
                "name": name,
                "layout": result.best_layout,
                "temperature": final_temperature,
                "metrics": result.best_cost.metrics,
            }
        )

    save_final_summary(case, final_results, output_dir / "optimizer_comparison_summary.png", "Optimizer comparison")
    pd.DataFrame(rows).to_csv(output_dir / "metrics.csv", index=False)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[compare] saved outputs to {output_dir}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare SA, GA, and RL optimizers.")
    parser.add_argument("--config", type=Path, default=Path("configs/optimizer_comparison.yaml"))
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
