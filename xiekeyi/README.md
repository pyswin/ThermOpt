# RL

有三套 RL floorplanning 方法，由 `reinforcement_learning.variant` 选择：

| variant | 实现文件 | 配置文件 |
| --- | --- | --- |
| `allreward` | `rl_test_0623_allreward.py` | `wl_only_comparison_allreward.yaml` |
| `effplace` | `rl_test_0623_EffPlace.py` | `wl_only_comparison_effplace.yaml` |
| `flexplanner` | `rl_test_0623_flexplanner.py` | `wl_only_comparison_flexplanner.yaml` |

## 使用

运行单个配置：

```bash
PYTHONPATH=src python -m xiekeyi.run_optimizer_comparison_rl_0620 --config xiekeyi/wl_only_comparison_allreward.yaml
```

或修改 `xiekeyi/run_v0_rl.sh` 中的 `--config` 后执行：

```bash
bash xiekeyi/run_v0_rl.sh
```

结果会写到 `outputs/<timestamp>_<experiment_name>/`，主要看：

- `summary.json`: 各 optimizer 的指标汇总
- `metrics.csv`: 表格版指标
- `rl_*.log`: RL 训练日志
- `final_layout_*.png`: 最终布局
- `cost_curve_*.png`: best curve

配置入口在 YAML 的 `reinforcement_learning` 段。常用字段包括
`variant`、`episodes`、`grid`、`placement_order`、`wire_mask_*`、
`reward_scale`、`terminal_reward_coef` 和 `elite_replay_*`。

## 三套方法

### allreward

grid placement/PPO。

特点：每步使用 wirelength increment reward，终局加入合法性和 HPWL 改善 reward，
并用 elite replay 强化当前 best episode。实现相对直接，不使用 tree frontier。
在 Case7 的现有实验里，这套通常最好，适合作为主结果。

### effplace

来源：EfficientPlace 迁移，包括 search tree/frontier。

特点：episode 可以从 frontier 恢复继续搜索，frontier 根据合法布局的 wirelength
做 backup；非法布局会用很大的惩罚 wirelength，避免低 HPWL 非法布局误导搜索。
这套更像 EfficientPlace + tree search 的 ablation，目前 Case7 上不如 `allreward` 稳定。

### flexplanner

来源：参考 FlexPlanner 的 grid canvas、position mask、wiremask 和混合 reward 思路。

特点：除 wirelength 外，还显式建模 overlap 相关 reward/penalty 和 selectable score，
支持更偏 FlexPlanner 的边界/重叠容忍配置。当前仍是 2D fixed-size chiplet placement，
不包含原 FlexPlanner 的 3D layer、aspect-ratio action 等完整能力。

## 注意

- 三套实现都把 grid action 映射回 ThermOpt 的连续坐标，rotation 固定为 0。
- `Objective` 主要用于返回统一 `CostResult` 和最终指标；RL selection 仍以 wirelength/合法性为主。
- `invalid_placement_penalty * len(order)` 这类随 case 改变的值由代码默认计算。
