# ThermOpt

ThermOpt 是一个 chiplet placement 实验仓库。当前主线目标是先把 **WL-only placement** 做稳，再接真实热仿真或训练好的热代理模型。

目前热仿真仍是启发式 Gaussian thermal field，只用于保持流程完整；和论文表格对比时，主要看 `TWL/m`。

## 目录结构

```text
configs/                  实验配置
outputs/                  实验输出，默认不提交
src/thermopt/data/         case 读取与生成
src/thermopt/layout/       布局对象、HPWL、合法性、可视化
src/thermopt/objective/    指标与 scalar objective
src/thermopt/optimizer/    placement 优化方法
src/thermopt/thermal/      当前启发式热模型，后续接 HotSpot/代理模型
src/thermopt/experiments/  实验入口
tests/                    回归测试
```

本地数据目录约定：

```text
external/ATPlace_pub/      ATPlace2.5D 官方公开 case 和加密 runner
pointwise/                 热代理模型训练数据
```

`external/`、`pointwise/`、`outputs/` 都被 `.gitignore` 忽略。

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

本地开发直接用：

```bash
PYTHONPATH=src pytest
```

## 数据

主 benchmark 使用 ATPlace2.5D 公开 case：

```bash
mkdir -p external
git clone --depth 1 https://github.com/Brilight/ATPlace_pub.git external/ATPlace_pub
```

loader 在 `src/thermopt/data/atplace.py`，读取 `.blocks/.nets/.power/.pl`，并修正为论文 Table 3 的 interposer 尺寸。`pointwise/` 只作为热模型训练数据，不作为主 placement benchmark。

## 当前优化方法

`src/thermopt/optimizer/` 里现在保留两个主要 WL-only 方法：

### `atplace`

文件：`src/thermopt/optimizer/atplace.py`

这是我们自己的可读实现，参考 ATPlace2.5D 的 WL-driven 思路：

1. 按 chiplet pair 聚合 net clump；
2. 用四方向 MILP 初始化，支持 0/90/180/270；
3. 连续优化 WL + density/overlap + outline；
4. sequence-pair 合法化/扰动；
5. 从所有阶段选择 best legal layout。

说明：ATPlace_pub 的核心 placement kernel 是 PyArmor 加密模块，不能直接合并成可维护源码。官方 runner 仍可作为 external baseline，但推荐 Gurobi 环境。

### `atmplace`

文件：`src/thermopt/optimizer/atmplace.py`

这是参考 ATMPlace 论文优化算法部分的 WL-only 版本，不包含热应力/warpage 目标。当前实现重点参考：

1. orientation-aware 初始化；
2. 连续解析优化；
3. density/overflow 权重调度；
4. legalization；
5. 大 case 默认避免全 pairwise MILP，改用更轻的 spectral/clump seed。

热、温度、warpage 以后可以作为 objective term 接入，但当前 benchmark 只看线长。

## 运行 WL Benchmark

默认跑 Case1-3，同时比较 `atplace` 和 `atmplace`：

```bash
PYTHONPATH=src python -m thermopt.experiments.run_optimizer_comparison --config configs/wl_benchmark.yaml
```

输出在：

```text
outputs/<timestamp>_wl_benchmark/
```

每个 case 会有：

```text
initial_layout.png
final_layout_atplace.png
final_layout_atmplace.png
metrics.csv
summary.json
optimizer_comparison_summary.png
```

如果要跑更多 case，修改 `configs/wl_benchmark.yaml` 里的 `case.cases`。Case9/Case10 对 MILP 类方法较重，建议先用 `atmplace` 的 spectral seed 或降低迭代预算。

## 旧流程

随机 case 的 V0 流程还保留：

```bash
bash scripts/run_v0.sh
```

它主要用于验证热模型、objective 和可视化链路，不用于论文线长对比。

## 测试

```bash
PYTHONPATH=src pytest
```

测试覆盖：

- ATPlace case loader；
- pin-offset HPWL 和四方向旋转；
- `atplace`/`atmplace` 小 case 合法布局；
- MILP、pointwise loader、旧 SA/GA/RL 基础流程。

## 当前限制

- 热仿真还没有接 HotSpot 或训练好的代理模型。
- `atmplace` 只参考优化算法，不实现论文里的 thermo-mechanical objective。
- 官方 ATPlace_pub 核心代码是加密 kernel；本仓库不直接复制其内部实现。
- SciPy HiGHS 不等价于 Gurobi，大 case 上 MILP 质量和 runtime 都会有差异。
