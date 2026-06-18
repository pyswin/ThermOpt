# optimizer 目录说明

这个目录放 placement 优化器。当前主线建议优先看 `atplace.py` 和 `atmplace.py`；其他文件主要是早期 baseline、工具函数或旧流程保留项。

## 当前主线方法

- `atplace.py`：参考 ATPlace2.5D 的 WL-only 优化流程。包含 MILP/clump 初始化、连续解析优化、density/outline penalty、sequence-pair legalization 和扰动逃逸。当前 WL benchmark 的主要方法之一。
- `atmplace.py`：参考 ATMPlace 优化算法部分的 WL-only 版本。不包含热应力/warpage objective。当前 WL benchmark 的主要方法之一。

## 公共工具和可选 baseline

- `sequence_pair.py`：sequence-pair 编码、退火搜索和 `decode_sequence_pair` legalization 工具。它既可以作为独立 baseline 运行，也被 `atplace.py` 和 `atmplace.py` 复用，当前不能直接删除。
- `milp_wl.py`：早期 WL-only MILP baseline。用 SciPy HiGHS MILP 建模非重叠、outline、旋转和二 pin HPWL。小 case 可用于 sanity check；大 case 会很慢，不是当前主线。

## 旧流程方法

- `simulated_annealing.py`：早期随机移动退火 baseline。用于验证 objective 和输出链路，不建议作为论文对比主结果。
- `genetic_algorithm.py`：早期遗传算法 baseline。用于比较随机启发式方法，不是当前主线。
- `rl_environment.py`：早期强化学习环境定义，提供状态、动作和 reward。
- `rl_policy.py`：早期策略梯度/RL baseline，依赖 `rl_environment.py`。

## 其他

- `__init__.py`：Python package 标记文件。

如果要继续清理目录，优先评估 `simulated_annealing.py`、`genetic_algorithm.py`、`rl_environment.py`、`rl_policy.py` 是否还需要保留为旧 baseline；不要删除 `sequence_pair.py`，除非同时重写 `atplace/atmplace` 的 legalization。
