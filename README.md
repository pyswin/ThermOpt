# ThermOpt

ThermOpt is a minimal reproducible framework for thermal-aware chiplet floorplanning. The V0 release keeps the stack intentionally small: heuristic thermal analysis, scalar objective functions, and simulated annealing.

The goal is to make the optimization loop easy to inspect and extend:

```text
case generation -> layout -> power/temperature map -> thermal cost -> optimizer -> figures and metrics
```

## Installation

```bash
git clone <your-repo-url>
cd thermopt
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For local development without installation, the scripts set `PYTHONPATH=src` automatically.

## Quick Start

```bash
bash scripts/run_v0.sh
```

The default run creates one timestamped folder under `outputs/` and executes three experiments:

- `wl_only`: optimize wirelength and legality only.
- `wl_tmax`: optimize wirelength, legality, and maximum temperature.
- `wl_topk`: optimize wirelength, legality, and top-5% mean temperature.

Expected output files include:

```text
initial_layout.png
initial_temperature.png
final_layout_wl_only.png
final_layout_wl_tmax.png
final_layout_wl_topk.png
final_temperature_wl_only.png
final_temperature_wl_tmax.png
final_temperature_wl_topk.png
cost_curve_wl_only.png
cost_curve_wl_tmax.png
cost_curve_wl_topk.png
metrics.csv
summary.json
config.yaml
```

`outputs/` is ignored by Git except for `outputs/.gitkeep`.

## Method Overview

V0 uses randomly generated chiplet cases with fixed outline constraints, optional random netlists, and per-chiplet power values. A layout is represented by each chiplet's `(x, y, rotation)` state.

The heuristic thermal simulator estimates a temperature field with Gaussian heat spreading:

```text
T(x, y) = Tamb + scale * sum_i P_i * exp(-d_i^2 / (2 * sigma_i^2))
```

This is not a physical replacement for HotSpot or a compact thermal solver. It is a lightweight oracle for validating the optimization loop: clustered high-power chiplets should produce higher hotspots, while spreading them should reduce hotspots.

## Objective Function

The optimizer minimizes:

```text
Cost = alpha * WL / WL0
     + beta  * Thermal / T0
     + gamma * outline_penalty
     + delta * overlap_penalty
```

Thermal modes:

- `tmax`: maximum grid temperature.
- `topk`: mean of the hottest `k` percent of grid cells.
- `threshold`: violation above a configured temperature limit.

`WL0` and `T0` are computed from the initial layout to keep terms roughly normalized.

## Configuration

Edit [`configs/v0_default.yaml`](configs/v0_default.yaml) to change the case size, thermal grid, objective weights, annealing schedule, or experiment list.

## Tests

```bash
PYTHONPATH=src pytest
```

## Project Structure

```text
src/thermopt/
  data/          random case generation
  layout/        data objects, geometry, rasterization, visualization
  thermal/       heuristic thermal simulator
  objective/     metrics and scalar objective
  optimizer/     simulated annealing
  experiments/   runnable V0 pipeline
```

## Roadmap

- V0: heuristic thermal analysis plus simulated annealing.
- V1: compact thermal solver or HotSpot wrapper.
- V2: generate thermal datasets and train U-Net/FNO/PNO surrogates.
- V3: embed learned thermal surrogates into optimization.
- V4: expose an RL environment with reward `-Cost`.
- V5: benchmark SA, RL, and learning-based thermal-aware placement methods.
