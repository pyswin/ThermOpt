# ThermOpt

ThermOpt is a chiplet floorplanning sandbox with two independent flows that share the same thermal backend:

- dataset generation: `case -> random layout/power/rotation -> thermal backend -> pointwise/grid/json`
- optimization: `case + initial layout -> wirelength + temperature -> cost -> optimizer updates layout`

The dataset path and the optimization path are separate consumers of `ThermalBackend`. They do not depend on each other in execution order.

The ATPlace benchmark cases are vendored under `external/ATPlace_pub/cases/`. The default HotSpot binary is vendored at `external/ATPlace_pub/thermal/hotspot`.

## Supported Optimizers

The current supported, non-neural optimizers are:

- `simulated_annealing`
- `genetic_algorithm`
- `sequence_pair`
- `milp_wl`
- `atplace`
- `atmplace`

These are the optimizers to use for the current project setup. `SA`, `atplace`, and `atmplace` are the main paths to keep in mind.

The repository still contains older experimental modules, but the quick-start path below does not rely on RL or other neural-network optimizers.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`scipy` is required for `milp_wl`, `atplace`, and `atmplace`. The default workflows do not require `torch`.

## Quick Start

1. Run a syntax smoke test:

```bash
python3 -m compileall src tests scripts
```

2. Run the default SA flow:

```bash
bash scripts/run_v0.sh
```

This uses `configs/v0_default.yaml`, which runs SA on synthetic random cases.

3. Run the ATPlace-family benchmark:

```bash
bash scripts/run_optimizer_comparison.sh
```

This uses `configs/wl_benchmark.yaml`, which runs the ATPlace-style optimizers on the vendored ATPlace cases.

4. Run the `atplace`-only benchmark:

```bash
bash scripts/run_atplace.sh
```

This uses `configs/atplace_benchmark.yaml`.

5. Run the `atmplace`-only benchmark:

```bash
bash scripts/run_atmplace.sh
```

This uses `configs/atmplace_benchmark.yaml`.

6. Generate a thermal dataset:

```bash
python3 scripts/generate_thermal_dataset.py \
  --case_dir external/ATPlace_pub/cases/Case1 \
  --output_dir /tmp/thermopt_dataset \
  --num_samples 10 \
  --variation_type random
```

## Available Configs

- `configs/v0_default.yaml`: SA on synthetic random cases.
- `configs/atplace_v0.yaml`: SA on ATPlace cases.
- `configs/atplace_benchmark.yaml`: standalone `atplace` benchmark.
- `configs/atmplace_benchmark.yaml`: standalone `atmplace` benchmark.
- `configs/wl_benchmark.yaml`: ATPlace-family benchmark for `atplace` and `atmplace`.
- `configs/optimizer_comparison.yaml`: older broad comparison config, kept for reference.

## Thermal Backend

The thermal backend is controlled by the `thermal` section in a config file:

```yaml
thermal:
  backend: hotspot   # or heuristic
  hotspot_binary: external/ATPlace_pub/thermal/hotspot
  hotspot_allow_fallback: true
```

- `backend: hotspot` uses the vendored HotSpot binary when available.
- `hotspot_allow_fallback: true` keeps the code runnable if HotSpot is missing.
- Set `hotspot_required: true` in a script or config if you want to force real HotSpot evaluation.
- `thermal.grid_size` is used as-is for HotSpot `grid_rows` and `grid_cols`; rectangular grids are supported.

## Objective

The objective combines wirelength and temperature:

```yaml
objective:
  alpha: 1.0
  beta: 1.0
  gamma: 50.0
  delta: 80.0
```

- `beta: 0.0` means wirelength-only optimization.
- `beta > 0.0` means temperature contributes to the total cost.

## Units And Rotation

- Internal layout units are `mm`.
- The HotSpot adapter converts to meters internally.
- Dataset generation uses `0` and `90` degree rotations.
- SA can use `0`, `90`, `180`, and `270` degree rotations, but `allow_rotate_move` controls whether the optimizer is allowed to use rotation moves.

## Outputs

Generated optimizer results are written under `outputs/`:

- `summary.json`
- `metrics.csv`
- `final_summary.png`
- `final_layout_*.png`
- `final_temperature_*.png`
- `cost_curve_*.png`

Dataset outputs are written separately as:

- `pointwise/sample_*.csv`
- `gridwise/sample_*.csv`
- `json/sample_*.json`
- `dataset_summary.json`

## Maintenance Boundary

- To replace the thermal solver later, keep the `ThermalBackend` interface and swap the implementation under `src/thermopt/thermal/`.
- To replace the optimizer later, keep the `Objective` callback signature and only change the code under `src/thermopt/optimizer/`.
- To move the dataset generator later, keep the ATPlace-style case loader and preserve the output schema in `pointwise/`, `gridwise/`, and `json/`.
