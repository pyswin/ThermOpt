# Therm-FM T 推理最小样例（case_all）

自包含的 Therm-FM T 模型推理 demo：给定输入样本（芯片功率场 + 网格坐标），输出预测的稳态温度场（开尔文）。`inputs/` 内置 10 个样本（case_all 的 10 种芯片配置各 1 个），可直接运行。

## 目录结构
```
thermfm_t_case_all_demo/
├── README.md
├── inference.py          # 推理脚本（自包含，核心，只需 torch/transformers/numpy）
├── model.py              # ScOT 模型定义（从 Therm-FM 拷出，只依赖 torch + transformers）
├── prepare_samples.py    # （可选）从原始数据集重新生成 inputs/，需要完整 Therm-FM 环境
├── model/                # Therm-FM T 权重（在 case_all 上微调）
│   ├── config.json
│   ├── pytorch_model.bin       (80 MB)
│   └── normalization_constants.json
├── inputs/               # 10 个输入样本（物理值）
│   └── sample_00.npz ... sample_09.npz
└── outputs/              # 10 个推理输出（开尔文），运行 inference.py 后生成
    └── sample_00.npz ... sample_09.npz
```

## 依赖
- Python 3.10
- torch（建议 2.0+）
- transformers（建议 ==4.29.2，开发所用版本；其他版本若 Swinv2 API 有变动可能需调整）
- numpy

安装：
```bash
pip install torch transformers numpy
```

## 运行
```bash
python inference.py          # 自动检测 GPU，没有则用 CPU
python inference.py --gpu    # 强制 GPU
python inference.py --cpu    # 强制 CPU
```
脚本读取 `inputs/*.npz`，对每个样本归一化 → 模型推理 → 反归一化，把预测温度场写到 `outputs/sample_NN.npz`。

## 数据格式

**输入** `inputs/sample_NN.npz`，字段 `input`，shape `3×64×64`，float32（物理值）：
| 通道 | 含义 | 单位 |
|------|------|------|
| 0 | chiplet_power（芯片功率密度） | W |
| 1 | grid_x（网格 x 坐标） | grid |
| 2 | grid_y（网格 y 坐标） | grid |

**输出** `outputs/sample_NN.npz`，字段 `prediction`，shape `1×64×64`（L_out=1 层，与模型原始输出一致），float32：预测稳态温度场，开尔文（K）。

`inference.py` 内部用 `model/normalization_constants.json` 做「归一化 → 推理 → 反归一化」，所以 input/output 都是真实物理量（功率/坐标 → 开尔文）。

## 10 个样本
取自 case_all 测试集（10 种芯片配置 × 200 测试样本，per-case 分层 80/20 切分）。`sample_00..09` 各对应一种 case（case1..case10），展示不同配置下「功率场 → 温度场」的映射。

## 模型
Therm-FM T = Poseidon-T 在 case_all（10 种 case、1 万样本）上微调的 3D-IC 稳态热仿真模型，scOT/Poseidon 架构（Swinv2 骨干）。模型定义见 `model.py`。
