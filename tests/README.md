# tests 目录说明

这个目录放 pytest 测试，覆盖数据加载、几何计算、objective、优化器和实验入口。

- `test_atplace_loader.py`：测试 ATPlace Bookshelf case loader，确认 `.blocks/.nets/.power/.pl` 能转换成内部 `FloorplanCase/Layout`。
- `test_pointwise_loader.py`：测试 pointwise CSV loader，确认热仿真训练数据能转成内部 case。
- `test_geometry.py`：测试基础几何逻辑，包括 HPWL、pin offset、旋转尺寸、outline 和 overlap。
- `test_cost.py`：测试 objective cost 组合逻辑，包括 wirelength、温度项和 penalty。
- `test_placement_methods.py`：测试当前主线 `atplace`、`atmplace` 在小 case 上能生成合法布局。
- `test_milp_wl.py`：测试早期 `milp_wl` baseline 在小 case 上能找到合法布局。
- `test_optimizer_comparison.py`：测试 `run_optimizer_comparison.py` 实验入口能生成 summary 和关键输出。
- `test_run_v0.py`：测试早期 V0 SA 流程能跑通。
- `test_genetic_algorithm.py`：测试早期 GA baseline 基础行为。
- `test_rl_environment.py`：测试早期 RL 环境的状态、动作和 step 逻辑。
- `test_rl_policy.py`：测试早期 RL baseline 能完成最小训练/rollout 流程。

常用命令：

```bash
PYTHONPATH=src pytest
```
