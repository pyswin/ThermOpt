# ThermOpt

ThermOpt is a minimal reproducible framework for thermal-aware chiplet floorplanning. The V0 release keeps the stack intentionally small: heuristic thermal analysis, scalar objective functions, and simulated annealing.

The goal is to make the optimization loop easy to inspect and extend:



# 目录说明
  configs/
  放实验配置，比如 chiplet 数量、outline、热仿真网格、目标函数权重、SA 参数。
   
  scripts/
  放用户入口脚本。现在主要是 scripts/run_v0.sh，一键跑 V0 实验。

  src/thermopt/data/
  负责生成输入 case。现在是随机 chiplet、power、netlist。

  src/thermopt/layout/
  负责布局数据结构、几何计算、rasterization、可视化。

  src/thermopt/thermal/
  热仿真模块。现在是启发式 Gaussian thermal field，后续 HotSpot、compact solver、U-Net/FNO 都应该接在这里。

  src/thermopt/objective/
  目标函数和指标模块。负责把 WL、temperature、penalty 合成 scalar cost。

  src/thermopt/optimizer/
  优化算法模块。现在是 simulated annealing，后续 RL、遗传算法、连续优化器都放这里。

  src/thermopt/experiments/
  主流程模块。负责读取 config、生成 case、调用 thermal/objective/optimizer、保存图片和 metrics。

  tests/
  基础测试和小型端到端测试。

  outputs/
  运行结果目录，只保留 .gitkeep，图片和 csv 不提交。

## 分工

  1. 热仿真负责人(hz)
     主要维护：

  - src/thermopt/thermal/
  - 必要时维护 src/thermopt/layout/rasterization.py

  目标是保证输入 case + layout，输出 temperature map。后续接 HotSpot、compact solver、surrogate 都不要影响
  optimizer。

  2. 目标函数负责人(YWK)
     主要维护：

  - src/thermopt/objective/cost.py
  - src/thermopt/objective/metrics.py
  - 部分 configs/v0_default.yaml 里的 objective 配置

  目标是定义 WL、Tmax、top-k、threshold、overlap、outline 等如何合成 cost。

  3. 优化方法负责人(XKY + HZ + YWK)
     主要维护：

  - src/thermopt/optimizer/
  - 必要时扩展 layout move set

  目标是只依赖 objective callback，不关心 thermal 具体怎么实现。

  4. 主流程负责人(HZ)
     主要维护：

  - src/thermopt/experiments/run_v0_sa.py
  - scripts/run_v0.sh
  - README.md
  - 输出格式、metrics、summary、可视化汇总

  目标是保证别人一键能跑、结果可复现、输出清楚。

  
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
final_summary.png
metrics.csv
summary.json
config.yaml
```

`outputs/` is ignored by Git except for `outputs/.gitkeep`.

To compare simulated annealing, genetic algorithm, and the lightweight RL policy optimizer:

```bash
bash scripts/run_optimizer_comparison.sh
```

This writes `final_layout_*`, `final_temperature_*`, `cost_curve_*`, `metrics.csv`, and `optimizer_comparison_summary.png` under a timestamped `outputs/*_optimizer_comparison/` folder. The comparison script also prints optimizer progress logs, including RL training episodes.

To run the V0 flow on ATPlace2.5D public benchmark cases, clone the public cases once and then run:

```bash
mkdir -p external
git clone --depth 1 https://github.com/Brilight/ATPlace_pub.git external/ATPlace_pub
PYTHONPATH=src python -m thermopt.experiments.run_v0_sa --config configs/atplace_v0.yaml
```

This uses `Case1`, `Case2`, and `Case3` from `external/ATPlace_pub/cases` as the main benchmark input. The loader reads each case's `.blocks`, `.nets`, and `.power` files into a `FloorplanCase`, then starts from a deterministic random initial layout. Thermal evaluation still uses the current heuristic simulator until HotSpot or a learned thermal model is wired into `src/thermopt/thermal/`.

The `pointwise/` CSVs are treated as thermal-model training data, not as the main placement benchmark. To compare those samples with the ATPlace cases before training or deployment:

```bash
PYTHONPATH=src python -m thermopt.experiments.compare_datasets
```

The comparison report is written to `outputs/dataset_comparison.json`.

For a WL-only baseline that is closer to the fixed-outline legalization problem, use the MILP optimizer on a small ATPlace case:

```bash
PYTHONPATH=src python -m thermopt.experiments.run_optimizer_comparison --config configs/atplace_wl_milp.yaml
```

The MILP model uses the same ATPlace-style pin-offset HPWL as `src/thermopt/layout/geometry.py`, binary non-overlap constraints, optional 0/90 rotation, and the current outline dimensions. It is intended as an exact or near-exact diagnostic baseline for small cases, not as the final scalable placement engine.

To reproduce the current ATPlace2.5D-style WL-only results for the first three public cases:

```bash
PYTHONPATH=src python -m thermopt.experiments.run_optimizer_comparison --config configs/atplace_wl_reproduce.yaml
```

For the larger follow-up run on Case4-Case10:

```bash
PYTHONPATH=src python -m thermopt.experiments.run_optimizer_comparison --config configs/atplace_wl_remaining.yaml
```

`configs/atplace_wl_remaining.yaml` is useful for Case4-Case8, but Case9 and Case10 are currently too large for the full pairwise MILP to finish reliably with SciPy HiGHS under the same time budget. The latest local Case1-Case8 comparison JSON is written to:

```text
outputs/20260617_203121_atplace_wl_remaining/case1_case8_wl_summary.json
```

Generated `outputs/` artifacts remain ignored by Git.

## Method Overview

V0 uses randomly generated chiplet cases with fixed outline constraints, optional random netlists, and per-chiplet power values. A layout is represented by each chiplet's `(x, y, rotation)` state.

The heuristic thermal simulator estimates a temperature field with Gaussian heat spreading:

```text
T(x, y) = Tamb + scale * sum_i P_i * exp(-d_i^2 / (2 * sigma_i^2))
```

This is not a physical replacement for HotSpot or a compact thermal solver. It is a lightweight oracle for validating the optimization loop: clustered high-power chiplets should produce higher hotspots, while spreading them should reduce hotspots.

### ATPlace2.5D-Style WL Optimizer

`src/thermopt/optimizer/atplace_wl.py` implements the current strongest WL-only flow. It is inspired by the WL-driven flow in the ATPlace2.5D paper, but it is not a byte-for-byte reproduction of the authors' implementation.

The implemented flow is:

1. Build a chiplet-pair clump model from the netlist, including ATPlace pin offsets.
2. Solve a fixed-outline MILP with four legal orientations: 0, 90, 180, and 270 degrees.
3. Refine the MILP layout with a continuous WL/density/outline objective.
4. Run a legal sequence-pair perturbation fallback.
5. Select the best legal layout across all phases.

This matches the paper's high-level ideas: MILP initialization, analytical placement, density/legality pressure, perturbation, and legalization. The main differences are practical:

- SciPy HiGHS is used instead of Gurobi.
- The MILP objective aggregates nets into chiplet-pair clumps for scalability.
- Thermal optimization is disabled in the WL-only configs because HotSpot and the learned thermal model are not wired into the main optimizer yet.
- Large Case9/Case10 runs need a more scalable analytical placement/legalization path or a commercial MILP solver.

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
  optimizer/     simulated annealing, genetic algorithm, and RL environment
  experiments/   runnable V0 pipeline
```

## Roadmap

- V0: heuristic thermal analysis plus simulated annealing.
- V1: compact thermal solver or HotSpot wrapper.
- V2: generate thermal datasets and train U-Net/FNO/PNO surrogates.
- V3: embed learned thermal surrogates into optimization.
- V4: expose an RL environment with reward `-Cost`.
- V5: benchmark SA, RL, and learning-based thermal-aware placement methods.
