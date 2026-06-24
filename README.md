# ThermOpt

ThermOpt is a chiplet floorplanning sandbox with two independent flows that share the same thermal backend:

- dataset generation: `case -> random layout/power/rotation -> thermal backend -> pointwise/grid/json`
- optimization: `case + initial layout -> wirelength + temperature -> cost -> optimizer updates layout`

The dataset path and the optimization path are separate consumers of `ThermalBackend`. They do not depend on each other in execution order.

The ATPlace benchmark cases are vendored under `external/ATPlace_pub/cases/`. HotSpot binaries are vendored under `external/ATPlace_pub/thermal/`.

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
  --output_dir outputs/thermopt_dataset
```

The detailed dataset-generation commands and option meanings are listed below.

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
  backend: hotspot   # hotspot, heuristic, or ai
  hotspot_binary: external/ATPlace_pub/thermal/hotspot
  hotspot_required: true
  hotspot_allow_fallback: false
```

- `backend: hotspot` is supported only on Linux and uses the vendored Linux x86-64 binary at `external/ATPlace_pub/thermal/hotspot` by default.
- macOS must not be used to generate HotSpot labels. Use `backend: heuristic` on macOS for local smoke tests until the AI backend replaces it, or run HotSpot generation on Linux.
- `backend: heuristic` is a deterministic analytic approximation based on chiplet power and distance. It is useful for local development, but it is not a HotSpot label source.
- `backend: hotspot` is the default for thermal runs. If HotSpot is missing or fails, the run raises an error.
- `backend: ai` is a reserved interface for a future AI thermal simulator. It currently raises `NotImplementedError`.
- `thermal.grid_size` is the target output resolution. HotSpot grid inputs are rounded up per axis to the next power of two, then resampled back to the requested size.
- HotSpot grid output can contain multiple `Layer N:` sections. The parser reads the chip layer (`Layer 4`) when present.

## Dataset Generation

`scripts/generate_thermal_dataset.py` builds thermal training data from an ATPlace-style case directory. It reads the case, creates layouts, applies optional randomization, runs the thermal backend, and writes `pointwise/`, `gridwise/`, `json/`, and `dataset_summary.json`.

Important behavior:

- `--num_samples` means the number of **successful** samples to produce.
- Failed attempts are retried until the requested number of successful samples is reached.
- Sample file names are contiguous: `sample_000000`, `sample_000001`, ...
- `pointwise` CSVs store grid coordinates, a chiplet label for the cell, the associated chiplet power, and the temperature.
- Grid cells outside any chiplet are labeled as `background` with `chiplet_power=0`.

### Common Commands

Generate 1000 successful samples with position randomization only:

```bash
python3 scripts/generate_thermal_dataset.py \
  --case_dir external/ATPlace_pub/cases/Case1 \
  --output_dir outputs/thermopt_dataset \
  --num_samples 1000 \
  --variation_type random \
  --no-randomize_power \
  --no-randomize_rotation
```

Keep layout fixed, only vary power:

```bash
python3 scripts/generate_thermal_dataset.py \
  --case_dir external/ATPlace_pub/cases/Case1 \
  --output_dir outputs/thermopt_dataset \
  --num_samples 1000 \
  --variation_type random \
  --no-randomize_position \
  --no-randomize_rotation
```

Keep layout fixed and generate a stable baseline dataset:

```bash
python3 scripts/generate_thermal_dataset.py \
  --case_dir external/ATPlace_pub/cases/Case1 \
  --output_dir outputs/thermopt_dataset \
  --num_samples 1000 \
  --variation_type fixed \
  --no-randomize_position \
  --no-randomize_power \
  --no-randomize_rotation
```

Generate a monotonic power sweep:

```bash
python3 scripts/generate_thermal_dataset.py \
  --case_dir external/ATPlace_pub/cases/Case1 \
  --output_dir outputs/thermopt_dataset \
  --num_samples 1000 \
  --variation_type grid
```

Generate with real HotSpot and fail if it is missing or fails:

```bash
python3 scripts/generate_thermal_dataset.py \
  --case_dir external/ATPlace_pub/cases/Case1 \
  --output_dir outputs/thermopt_dataset \
  --num_samples 1000 \
  --backend hotspot \
  --hotspot_required \
  --no-hotspot_allow_fallback
```

### Option Reference

| Option | Meaning |
| --- | --- |
| `--case_dir` | Input ATPlace-style case directory containing `.blocks`, `.nets`, `.power`, and `.pl`. |
| `--output_dir` | Output directory for dataset files. |
| `--num_samples` | Target number of successful samples. The generator retries until this count is reached. |
| `--variation_type` | `random`, `fixed`, or `grid`. Controls how layouts and powers are generated. |
| `--save_formats` | Comma-separated output formats: `pointwise`, `gridwise`, `json`. |
| `--config_name` | Case config file name, usually `reproduce.json`. |
| `--config_mode` | Case config mode, `thermal` or `wl`. |
| `--use_case_config` / `--no-use_case_config` | Whether to read extra case settings from the case directory. |
| `--unit_scale` | Scale factor that converts case units into mm. |
| `--initial_layout` | Initial layout source, `pl` or `random`. |
| `--min_gap` | Minimum allowed gap between chiplets, in mm. |
| `--randomize_position` / `--no-randomize_position` | Enable or disable position randomization. |
| `--randomize_power` / `--no-randomize_power` | Enable or disable power randomization. |
| `--randomize_rotation` / `--no-randomize_rotation` | Enable or disable rotation randomization. |
| `--power_additive_fraction` | Adds absolute power perturbation to improve low-power coverage. |
| `--power_dropout_prob` | Probability of applying a low-power dropout state. |
| `--power_sleep_ratio` | Power ratio used for the dropout state. |
| `--power_shutdown_prob` | Probability of forcing a chiplet to 0 W. |
| `--min_power_density` / `--max_power_density` | Lower and upper power-density bounds used to clamp random power. |
| `--tdp_limit` / `--tdp_limit_ratio` | Soft total-power cap for the whole chip. |
| `--backend` | Thermal backend: Linux `hotspot`, development-only `heuristic`, or reserved `ai`. |
| `--hotspot_binary` | Path to the Linux HotSpot executable. Defaults to the vendored Linux x86-64 binary. |
| `--hotspot_required` / `--no-hotspot-required` | Fail if the HotSpot binary is missing. Defaults to required. |
| `--hotspot_allow_fallback` / `--no-hotspot-allow-fallback` | Deprecated compatibility option. HotSpot failures raise errors; no heuristic fallback is used. |
| `--grid_size NX NY` | Target thermal grid resolution. |
| `--ambient` | Ambient temperature, in Celsius. |
| `--scale` | Power-to-temperature scaling factor used by the thermal backend. |
| `--sigma_factor` | Reserved thermal configuration parameter. |
| `--thermal_threshold` | Optional thermal threshold used by HotSpot config generation. |
| `--work_dir` | Workspace for HotSpot temporary files. |
| `--seed` | Random seed. |

### Practical Notes

- If you want a dataset for temperature-field training, keep `pointwise` and `json` in `--save_formats`.
- If you only care about the raster temperature map, `gridwise` is the smallest output.
- `background` cells in `pointwise` are intentional. They represent grid locations outside any chiplet footprint.
- `grid_size` in the dataset command is the requested output resolution, not the internal HotSpot grid. HotSpot may round its internal grid up and then resample back.

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
