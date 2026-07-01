# ThermOpt

芯粒（Chiplet）热感知布局优化实验平台。集成 ATPlace（ICCAD 2024）的三阶段优化框架，并在梯度优化阶段引入 ScOT（FNO 神经算子）作为可微分热仿真代理模型，实现位置→温度梯度的端到端回传。最终温度由 HotSpot 真实仿真验证。

---

## 目录结构

```
ThermOpt/
├── src/thermopt/
│   ├── optimizer/atplace.py          # ATPlace 三阶段优化器（含热损失与 WL 软约束）
│   ├── thermal/
│   │   ├── grad_thermal.py           # ScOT 可微分推理 + 软光栅化
│   │   ├── hotspot.py                # HotSpot 真实仿真后端
│   │   ├── thermfm.py                # ThermFM/ScOT 预测后端（非梯度评估用）
│   │   └── thermfm_t_case_all_demo/  # ScOT 模型权重
│   └── ...
├── scripts/
│   ├── thermal_optimize.py           # 热优化主脚本（全力降温，WL 无约束）
│   ├── thermal_tradeoff.py           # WL-Thermal 权衡分析脚本（多权重/多约束）
│   └── hotspot_validate.py           # HotSpot 真实仿真验证脚本
├── external/ATPlace_pub/
│   ├── cases/                        # ATPlace benchmark cases（Case1–Case10）
│   └── thermal/hotspot               # HotSpot Linux x86-64 binary
└── atplace/
    ├── 20260627_163106_milp150s/     # MILP 基准解（ATPlace 第一阶段，150s 时限）
    └── thermal_runs/
        └── 20260628_004144_maxT/     # 热优化最终结果（含 HotSpot 验证）
```

---

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# ScOT 梯度推理需要 torch（CUDA 推荐）和 transformers 4.35–4.49
pip install "transformers>=4.35,<4.50"
```

> `transformers>=5.0` 要求 `torch>=2.6`，与当前环境（torch 2.5.1）不兼容，请固定版本范围。

---

## 核心流程

### 热优化工作流

```
MILP 初始解（ATPlace 第一阶段，外部调用）
        ↓
梯度优化（ATPlace 第二阶段，ThermOpt 扩展）
  总损失 = wl_weight    × HPWL(xy)
         + thermal_weight × ScOT_loss(xy)      ← 新增
         + density_weight  × 重叠惩罚
         + outline_weight  × 越界惩罚
         [+ wl_budget_weight × relu(HPWL - budget)²]  ← 可选 WL 软约束

  ScOT 梯度链路（可微分）：
    xy [N,2, float64, CPU]
    → .float().to(cuda)
    → 软光栅化（Sigmoid mask）→ 功率图 [64×64]
    → ScOT 神经算子（FNO）前向推理
    → Tmax 或 Tmax50 标量
    → .double().cpu() → 汇入总损失 → .backward()
        ↓
HotSpot 真实仿真验证最终布局温度
```

### 热目标

| 目标 | 公式 | 特点 |
|------|------|------|
| `tmax` | `temp_c.max()` | 直接优化峰值温度 |
| `tmax50` | `temp_c.flatten().topk(50).mean()` | 梯度更平滑，针对热点分布 |

### WL 软约束

```python
WL_penalty = wl_budget_weight × relu(smooth_HPWL − initial_HPWL × wl_budget_factor)²
```

- `wl_budget_factor=0`：无约束（全力降温）
- `wl_budget_factor=1.42`：线长最多允许增长 42%

---

## 运行脚本

### 1. 热优化（全力降温）

```bash
python3 scripts/thermal_optimize.py \
    --cases Case3 Case5 Case6 Case7 Case8 \
    --weight 10000 \
    --tag my_run
```

**参数说明：**

| 参数 | 说明 |
|------|------|
| `--cases` | case 列表，推荐用线长偏差 <15% 的：Case3/5/6/7/8 |
| `--weight` | `thermal_weight`，默认 10000（与 `wl_weight=1` 比值约 1:10000） |
| `--tag` | 输出目录后缀 |

结果保存至 `atplace/thermal_runs/{timestamp}_{tag}/`，每个 case 下含 `tmax/` 和 `tmax50/` 子目录，各有 `summary.json`（含最终芯粒坐标、温度、运行时间）。

### 2. WL-Thermal 权衡分析

```bash
python3 scripts/thermal_tradeoff.py \
    --cases Case3 Case5 \
    --weights 2000 5000 10000 \
    --tag tradeoff_exp
```

对每组 `(case, thermal_weight, WL预算, 热目标)` 组合独立运行，输出完整对比表。tight/loose 两档 WL 预算在脚本内的 `TIGHT_BUDGET` 字典中按 case 配置。

### 3. HotSpot 真实仿真验证

```bash
python3 scripts/hotspot_validate.py
```

读取 `atplace/thermal_runs/20260628_004144_maxT/` 下的优化结果，对初始布局和优化后布局各运行一次 HotSpot，输出温度对比表并保存至 `hotspot_validation.json`。

> HotSpot 每次仿真约 1–3 分钟，5 个 case × 3 次 = 约 20–30 分钟。

---

## 关键参数参考

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `refine_steps` | 300 | 梯度优化迭代步数 |
| `learning_rate` | 0.05 | Adam 学习率（mm/step） |
| `wl_weight` | 1.0 | 线长损失权重 |
| `thermal_weight` | 10000 | 热损失权重 |
| `density_weight` | 5000 | 芯粒重叠惩罚权重 |
| `outline_weight` | 20000 | 越界惩罚权重 |
| `wl_budget_factor` | 0.0 | WL 软约束倍数（0=无约束） |
| `wl_budget_weight` | 1e5 | WL 软约束惩罚系数 |
| `thermal_mode` | `tmax` | 热目标：`tmax` 或 `tmax50` |
| `thermal_topk` | 50 | `tmax50` 模式下取前 k 个热点 |

---

## 实验结果（tw=10000，无 WL 约束，HotSpot 验证）

有效 case（ThermOpt HPWL 与 ATPlace TWL 偏差 <15%）：Case3、Case5、Case6、Case7、Case8。
Case1/2/4 存在坐标解析偏差，暂不纳入；Case9/10 未测试。

以下为 HotSpot 真实仿真结果（`tmax` 目标，起点为 MILP 初始布局）：

| Case | 初始 Tmax | 优化后 Tmax | ΔTmax | 初始 Tmax50 | 优化后 Tmax50 | ΔTmax50 | WL 变化 |
|------|-----------|------------|-------|------------|--------------|---------|---------|
| Case3 | 102.7°C | 100.8°C | **-1.9°C**  | 102.5°C | 100.6°C | -1.9°C | +41% |
| Case5 | 130.0°C | 124.3°C | **-5.6°C**  | 128.4°C | 124.2°C | -4.2°C | +13% |
| Case6 | 100.6°C |  88.6°C | **-12.0°C** |  98.0°C |  88.1°C | -9.9°C | +19% |
| Case7 |  71.3°C |  63.4°C | **-7.9°C**  |  69.2°C |  63.3°C | -5.9°C |  -1% |
| Case8 |  61.6°C |  59.7°C | **-1.9°C**  |  60.9°C |  59.5°C | -1.4°C |  +3% |

**ScOT 预测误差**：ScOT 作为代理模型，预测误差因 case 而异（-28°C 到 +7°C），但梯度方向基本正确，优化后各 case 温度均有实际降低。Case5 的 ScOT 初始温度预测偏低约 28°C（102°C vs HotSpot 130°C），说明该布局已偏离 ScOT 训练分布。

完整数值（含 tmax50 目标及 ScOT 预测误差列）见 `atplace/thermal_runs/20260628_004144_maxT/hotspot_validation.json`。

---

## 单元约定

| 量 | 单位 | 备注 |
|----|------|------|
| 布局坐标 | mm | 内部表示；layout.json 存储 μm，读取时 ×0.001 |
| 线长（ATPlace） | m | `twl_m = hpwl / 1e6` |
| 线长（ThermOpt） | m | `thermopt_wl_m = hpwl() / 1e3` |
| 温度 | °C | HotSpot 输出 Kelvin，减 273.15；ScOT 同 |
| ScOT 输入网格 | 64×64 | 坐标范围 0 到 `outline_width/height`（mm） |

---

## 热仿真后端

```yaml
thermal:
  backend: hotspot        # hotspot / thermfm / heuristic
  hotspot_binary: external/ATPlace_pub/thermal/hotspot
```

| 后端 | 用途 | 平台限制 |
|------|------|---------|
| `hotspot` | 真实仿真，最终验证 | Linux x86-64 |
| `thermfm` | ScOT 神经算子，梯度优化阶段 | 需要 CUDA（推荐） |
| `heuristic` | 解析近似，本地快速调试 | 全平台 |

---

## 注意事项

- **ScOT 泛化**：对偏离训练分布较远的布局，ScOT 预测与 HotSpot 可能有 ±5°C 左右偏差；热优化后须用 HotSpot 验证。
- **Case7/Case8**：初始温度 <65°C，热优化改善幅度有限（<2°C），温度梯度信号弱。
- **线长代价**：`thermal_weight=10000` 时热目标主导，Case3 等高温 case 线长可能上涨 40% 以上。
- **模型缓存**：ScOT 通过 `_MODEL_CACHE` 在进程内只加载一次，多 case 连续跑无需重复加载。

## Dataset Generation

Important behavior:

- `--num_samples` means the number of **successful** samples to produce.
- Failed attempts are retried until the requested number of successful samples is reached.
- Sample file names are contiguous: `sample_000000`, `sample_000001`, ...
- `pointwise` CSVs store grid coordinates, a chiplet label for the cell, the associated chiplet power, and the temperature.
- Grid cells outside any chiplet are labeled as `background` with `chiplet_power=0`.
- You can post-process existing `pointwise` CSVs to add geometry channels for surrogate training:
  - `occupancy_mask`: 1 inside any chiplet footprint, 0 otherwise.
  - `edge_mask`: 1 near chiplet boundaries, 0 otherwise.
  - `coord_x_norm`, `coord_y_norm`: normalized grid coordinates in `[0, 1]`.
  - The augmentation script preserves the original pointwise columns and does not write empty values.

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

Augment an existing dataset in place or into a separate directory:

```bash
python3 scripts/augment_pointwise_features.py \
  --dataset_dir outputs/thermopt_dataset/case1
```

Write the augmented CSVs to another directory:

```bash
python3 scripts/augment_pointwise_features.py \
  --dataset_dir outputs/thermopt_dataset/case1 \
  --output_dir outputs/thermopt_dataset/case1_augmented
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
- Dataset generation always uses HotSpot as the golden thermal backend. Therm-FM and U-FNO are only available on the optimizer path.
- `grid_size` in the dataset command is the requested output resolution, not the internal HotSpot grid. HotSpot may round its internal grid up and then resample back.
- For FNO-style surrogate training, `occupancy_mask` and `edge_mask` usually help most with geometry recovery; `coord_x_norm/coord_y_norm` provide the positional cue without introducing extra coordinate scales.

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
