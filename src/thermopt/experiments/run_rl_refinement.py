"""Experiment: RL local search refinement from atplace/atmplace baseline.

Pipeline per case:
  1. Run baseline optimizer (atplace or atmplace) to get a strong initial layout.
  2. Run RL local search starting from that layout.
  3. Report wirelength comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import yaml

from thermopt.data.inputs import CaseInput
from thermopt.experiments.run_v0_sa import load_inputs
from thermopt.layout.geometry import hpwl
from thermopt.layout.visualization import save_cost_curve, save_layout_figure
from thermopt.objective.cost import Objective
from thermopt.optimizer import atplace, atmplace
from thermopt.optimizer.rl_local_search import optimize as rl_optimize
from thermopt.thermal.backend import build_thermal_backend


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_output_dir(config: dict) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = config.get("experiment_name", "rl_refinement")
    output_dir = Path(config.get("output_root", "outputs")) / f"{stamp}_{name}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def run_single_case(
    config: dict,
    config_path: Path,
    case_input: CaseInput,
    output_dir: Path,
    seed: int,
) -> dict:
    case = case_input.case
    initial_layout = case_input.layout
    thermal_backend = build_thermal_backend(
        case, config["thermal"], work_dir=output_dir / "_thermal",
    )
    objective = Objective(
        case, config["thermal"], config["objective"],
        initial_layout, thermal_backend=thermal_backend,
    )

    save_layout_figure(
        case, initial_layout, output_dir / "random_initial.png", "Random initial",
    )

    # --- Step 1: baseline optimizer ---
    baseline_name = str(config.get("baseline_optimizer", "atplace"))
    baseline_config = config.get(baseline_name, {})
    if baseline_name == "atmplace":
        baseline_fn = atmplace.optimize
    else:
        baseline_fn = atplace.optimize

    print(f"  [{case_input.name}] running baseline={baseline_name} ...")
    t0 = time.perf_counter()
    baseline_result = baseline_fn(
        case, initial_layout, objective, baseline_config, seed,
    )
    baseline_time = time.perf_counter() - t0
    baseline_layout = baseline_result.best_layout
    baseline_cost = baseline_result.best_cost
    baseline_wl = float(hpwl(case, baseline_layout))
    print(
        f"  [{case_input.name}] baseline done  "
        f"wl={baseline_wl:.2f}  cost={baseline_cost.total:.4f}  "
        f"time={baseline_time:.1f}s"
    )

    save_layout_figure(
        case, baseline_layout,
        output_dir / f"baseline_{baseline_name}.png",
        f"Baseline: {baseline_name}",
    )
    save_cost_curve(
        baseline_result.best_curve,
        output_dir / f"cost_curve_{baseline_name}.png",
        f"Cost curve: {baseline_name}",
    )

    # --- Step 2: RL local search refinement ---
    rl_config = config.get("rl_local_search", {})
    print(f"  [{case_input.name}] running RL local search refinement ...")
    t0 = time.perf_counter()
    rl_result = rl_optimize(
        case, baseline_layout, objective, rl_config, seed + 5000,
    )
    rl_time = time.perf_counter() - t0
    rl_wl = rl_result.final_wl
    print(
        f"  [{case_input.name}] RL done  "
        f"wl={rl_wl:.2f}  cost={rl_result.best_cost.total:.4f}  "
        f"accepted={rl_result.accepted_ratio:.2%}  time={rl_time:.1f}s"
    )

    save_layout_figure(
        case, rl_result.best_layout,
        output_dir / "rl_refined.png", "RL refined",
    )
    save_cost_curve(
        rl_result.best_curve,
        output_dir / "cost_curve_rl.png", "Cost curve: RL local search",
    )
    if rl_result.training_loss_curve:
        save_cost_curve(
            rl_result.training_loss_curve,
            output_dir / "rl_loss_curve.png", "DQN training loss",
        )

    # --- Step 3: comparison ---
    wl_delta = rl_wl - baseline_wl
    wl_pct = wl_delta / max(baseline_wl, 1e-9) * 100.0
    improved = wl_delta < -1e-6
    print(
        f"  [{case_input.name}] WL change: {baseline_wl:.2f} -> {rl_wl:.2f}  "
        f"({wl_pct:+.2f}%)  {'IMPROVED' if improved else 'no improvement'}"
    )

    summary = {
        "case": case_input.name,
        "num_chiplets": len(case.chiplets),
        "num_nets": len(case.nets),
        "baseline_optimizer": baseline_name,
        "baseline_wl": baseline_wl,
        "baseline_cost": float(baseline_cost.total),
        "baseline_time_sec": baseline_time,
        "rl_wl": rl_wl,
        "rl_cost": float(rl_result.best_cost.total),
        "rl_time_sec": rl_time,
        "rl_accepted_ratio": rl_result.accepted_ratio,
        "rl_total_steps": rl_result.attempted_moves,
        "wl_delta": wl_delta,
        "wl_change_pct": wl_pct,
        "improved": improved,
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def run(config_path: Path) -> Path:
    config = load_config(config_path)
    output_dir = make_output_dir(config)
    shutil.copy2(config_path, output_dir / "config.yaml")

    seed = int(config.get("seed", 17))
    inputs = load_inputs(config, seed)
    summaries: list[dict] = []

    for index, case_input in enumerate(inputs):
        case_dir = output_dir
        if len(inputs) > 1:
            case_dir = output_dir / case_input.name
            case_dir.mkdir(parents=True, exist_ok=False)
        print(f"[rl_refinement] === {case_input.name} ===")
        summary = run_single_case(
            config, config_path, case_input, case_dir, seed + index * 1000,
        )
        summaries.append(summary)

    # --- aggregate ---
    if len(summaries) > 1:
        with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump({"config": str(config_path), "cases": summaries}, f, indent=2)

        fieldnames = list(summaries[0].keys())
        with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in summaries:
                writer.writerow(row)

    print(f"[rl_refinement] saved outputs to {output_dir}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RL local search refinement from baseline optimizer.",
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/rl_refinement.yaml"),
    )
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
