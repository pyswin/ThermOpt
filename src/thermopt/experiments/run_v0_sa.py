from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import yaml

from thermopt.data.atplace import load_atplace_cases
from thermopt.data.case_generator import generate_random_case, random_initial_layout
from thermopt.data.inputs import CaseInput
from thermopt.data.pointwise import load_pointwise_cases
from thermopt.layout.visualization import save_cost_curve, save_final_summary, save_layout_figure, save_temperature_figure
from thermopt.objective.cost import Objective
from thermopt.objective.metrics import collect_metrics
from thermopt.optimizer.simulated_annealing import optimize
from thermopt.thermal.backend import build_thermal_backend


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


def load_inputs(config: dict, seed: int) -> list[CaseInput]:
    case_config = config["case"]
    source = str(case_config.get("source", "random")).lower()
    if source == "atplace":
        return load_atplace_cases(case_config, seed)
    if source == "pointwise":
        return load_pointwise_cases(case_config)
    if source != "random":
        raise ValueError(f"Unknown case source: {source}")

    case = generate_random_case(case_config, seed)
    initial_layout = random_initial_layout(case, seed + 1)
    return [CaseInput("random", case, initial_layout, Path(""))]


def run_single_case(config: dict, config_path: Path, case_input: CaseInput, output_dir: Path, seed: int) -> dict:
    case = case_input.case
    initial_layout = case_input.layout
    thermal_backend = build_thermal_backend(case, config["thermal"], work_dir=output_dir / "_thermal" / case_input.name)
    save_layout_figure(case, initial_layout, output_dir / "initial_layout.png", "Initial layout")
    has_thermal_experiment = any(
        abs(float(deep_update(config, experiment)["objective"].get("beta", 1.0))) > 1e-12
        for experiment in config.get("experiments", [])
    )
    if has_thermal_experiment:
        initial_temperature = thermal_backend.simulate(case, initial_layout)
        save_temperature_figure(initial_temperature, output_dir / "initial_temperature.png", "Initial temperature")

    rows: list[dict] = []
    final_results: list[dict] = []
    summary: dict = {
        "seed": seed,
        "config": str(config_path),
        "output_dir": str(output_dir),
        "case": case_input.name,
        "source_path": str(case_input.source_path) if str(case_input.source_path) else None,
        "num_chiplets": len(case.chiplets),
        "num_nets": len(case.nets),
        "thermal": {
            "requested_backend": str(config["thermal"].get("backend", "hotspot")),
            "runtime_mode": getattr(thermal_backend, "runtime_mode", thermal_backend.name),
        },
        "experiments": {},
    }

    for index, experiment in enumerate(config.get("experiments", [])):
        exp_name = experiment["name"]
        exp_config = deep_update(config, experiment)
        exp_backend = thermal_backend if exp_config["thermal"] == config["thermal"] else build_thermal_backend(
            case,
            exp_config["thermal"],
            work_dir=output_dir / "_thermal" / exp_name,
        )
        objective = Objective(case, exp_config["thermal"], exp_config["objective"], initial_layout, thermal_backend=exp_backend)

        started = time.perf_counter()
        result = optimize(
            case=case,
            initial_layout=initial_layout,
            objective=objective,
            config=exp_config["optimizer"],
            seed=seed + 100 + index,
        )
        runtime = time.perf_counter() - started

        final_temperature = exp_backend.simulate(case, result.best_layout)
        final_metrics = collect_metrics(
            case,
            result.best_layout,
            final_temperature,
            thermal_mode=exp_config["objective"].get("thermal_mode", "topk"),
            topk_percent=float(exp_config["objective"].get("topk_percent", 0.05)),
            temperature_limit=float(exp_config["objective"].get("temperature_limit", 85.0)),
        )
        report_metrics = dict(result.best_cost.metrics)
        report_metrics.update(final_metrics)
        report_metrics["total_cost"] = float(result.best_cost.total)
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
            "thermal_backend": getattr(exp_backend, "runtime_mode", exp_backend.name),
            **report_metrics,
        }
        rows.append(row)
        final_results.append(
            {
                "name": exp_name,
                "layout": result.best_layout,
                "temperature": final_temperature,
                "metrics": report_metrics,
            }
        )
        summary["experiments"][exp_name] = row

    if final_results:
        save_final_summary(case, final_results, output_dir / "final_summary.png")
    if rows:
        fieldnames: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
        with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    else:
        (output_dir / "metrics.csv").write_text("", encoding="utf-8")
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def run(config_path: Path) -> Path:
    config = load_config(config_path)
    output_dir = make_output_dir(config)
    shutil.copy2(config_path, output_dir / "config.yaml")

    seed = int(config.get("seed", 0))
    inputs = load_inputs(config, seed)
    summaries = {}
    for index, case_input in enumerate(inputs):
        case_output_dir = output_dir
        if len(inputs) > 1:
            case_output_dir = output_dir / case_input.name
            case_output_dir.mkdir(parents=True, exist_ok=False)
        summaries[case_input.name] = run_single_case(
            config,
            config_path,
            case_input,
            case_output_dir,
            seed + index * 1000,
        )

    if len(inputs) > 1:
        with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump({"config": str(config_path), "output_dir": str(output_dir), "cases": summaries}, f, indent=2)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ThermOpt V0 simulated annealing experiments.")
    parser.add_argument("--config", type=Path, default=Path("configs/v0_default.yaml"))
    args = parser.parse_args()
    output_dir = run(args.config)
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
