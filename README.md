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
