from __future__ import annotations

import argparse
import json
import shutil
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from thermopt.data.case_generator import generate_random_case, random_initial_layout
from thermopt.layout.visualization import save_cost_curve, save_final_summary, save_layout_figure, save_temperature_figure
from thermopt.objective.cost import Objective
from xiekeyi.rl_test_0623_EffPlace import optimize
from thermopt.thermal.heuristic import simulate_temperature


def deep_update(base: dict, updates: dict) -> dict:
    result = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_output_dir(config: dict) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = config.get("experiment_name", "v0_sa")
    output_dir = Path(config.get("output_root", "outputs")) / f"{stamp}_{name}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def run(config_path: Path) -> Path:
    config = load_config(config_path)
    output_dir = make_output_dir(config)
    shutil.copy2(config_path, output_dir / "config.yaml")

    seed = int(config.get("seed", 0))
    case = generate_random_case(config["case"], seed)
    initial_layout = random_initial_layout(case, seed + 1)
    initial_temperature = simulate_temperature(case, initial_layout, config["thermal"])
    save_layout_figure(case, initial_layout, output_dir / "initial_layout.png", "Initial layout")
    save_temperature_figure(initial_temperature, output_dir / "initial_temperature.png", "Initial temperature")

    rows: list[dict] = []
    final_results: list[dict] = []
    summary: dict = {
        "seed": seed,
        "config": str(config_path),
        "output_dir": str(output_dir),
        "experiments": {},
    }

    for index, experiment in enumerate(config.get("experiments", [])):
        exp_name = experiment["name"]
        exp_config = deep_update(config, experiment)
        objective = Objective(case, exp_config["thermal"], exp_config["objective"], initial_layout)

        started = time.perf_counter()
        result = optimize(
            case=case,
            initial_layout=initial_layout,
            objective=objective,
            config=exp_config["reinforcement_learning"],
            seed=seed + 100 + index,
        )
        runtime = time.perf_counter() - started

        final_temperature = simulate_temperature(case, result.best_layout, exp_config["thermal"])
        save_layout_figure(case, result.best_layout, output_dir / f"final_layout_{exp_name}.png", f"Final layout: {exp_name}")
        save_temperature_figure(
            final_temperature,
            output_dir / f"final_temperature_{exp_name}.png",
            f"Final temperature: {exp_name}",
        )
        save_cost_curve(result.best_curve, output_dir / f"cost_curve_{exp_name}.png", f"Cost curve: {exp_name}")

        row = {
            "experiment": exp_name,
            "runtime_sec": runtime,
            "accepted_ratio": result.accepted_ratio,
            **result.best_cost.metrics,
        }
        rows.append(row)
        final_results.append(
            {
                "name": exp_name,
                "layout": result.best_layout,
                "temperature": final_temperature,
                "metrics": result.best_cost.metrics,
            }
        )
        summary["experiments"][exp_name] = row

    if final_results:
        save_final_summary(case, final_results, output_dir / "final_summary.png")
    pd.DataFrame(rows).to_csv(output_dir / "metrics.csv", index=False)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ThermOpt V0 simulated annealing experiments.")
    parser.add_argument("--config", type=Path, default=Path("configs/v0_default.yaml"))
    args = parser.parse_args()
    output_dir = run(args.config)
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
