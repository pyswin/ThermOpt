# experiments 目录说明

这个目录放可以从命令行运行的实验脚本，负责加载配置、调用优化器、保存图片和 summary。

- `run_optimizer_comparison.py`：当前 WL benchmark 主入口。根据 config 中出现的 optimizer 配置块运行对应方法，例如 `atplace`、`atmplace`，并输出每个 case 的 layout 图、temperature 图、cost curve、`metrics.csv` 和 `summary.json`。
- `compare_datasets.py`：数据集差异检查脚本。用于比较 ATPlace case 和 pointwise 热仿真训练数据的规模、面积利用率、功耗范围、netlist 是否存在等，帮助判断热代理模型后续能否迁移使用。
- `run_v0_sa.py`：早期 V0 流程入口。主要用于随机 case、旧 SA/GA/RL 链路、热模型和可视化的 smoke test，不作为当前论文 WL 对比主流程。

当前主流程建议使用：

```bash
PYTHONPATH=src python -m thermopt.experiments.run_optimizer_comparison --config configs/wl_benchmark.yaml
```
