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
