# RTX 5060 Ti 8GB 训练加速

2026-11-17 在本机检测到 NVIDIA GeForce RTX 5060 Ti（8GB）、PyTorch 2.11.0+cu128、CUDA 12.8。现有环境已可运行 CUDA，无需重装依赖。

## 默认运行参数

| 配置 | 训练 batch | 验证/校准/阈值选择/测试 batch |
| --- | ---: | ---: |
| `configs/train.yaml`，多模态 | 128 | 256 |
| `configs/vital_thresholds.yaml`，体征 | 256 | 512 |

两者使用自动 BF16/FP16、`deterministic: false` 和 `allow_tf32: true`。默认学习率保持不变，仍以验证 AUPRC 选择模型、以独立验证集最大 F1 选择阈值。NEWS2 是固定规则计算；XGBoost 仍采用 CPU 外存训练，以上 CUDA 设置针对四种神经网络。

`eval_batch_size: 0` 沿用训练 batch；大批量文本输入发生显存不足时可分别减小训练、验证和文本编码 batch。`deterministic: true` 恢复确定性算法；`allow_tf32: false` 关闭 TF32。关闭 AMP 时需同时关闭 TF32 才是完整 FP32 矩阵计算。大批量会减少每轮更新次数，速度测试不说明模型质量提高，需要重新评价完整训练后的指标。

代码优化包括：训练不构造无用的患者元数据字典；导出按批读取元数据；体征标准化 GPU 常量只初始化一次；训练和验证使用独立 batch；保存实际配置以及每轮训练时间、吞吐和验证时间。保留输入有效性检查、患者隔离、所有训练样本及数值标准化规则。

## 本机合成输入测速

测试输入为 `[batch, 120, 20]` 的体征窗口，使用默认模型宽度、BF16；每组预热 4 步、计时 12 步，计时边界同步 CUDA。包含锁页 CPU 到 GPU 传输、前向、损失、反向、梯度裁剪和优化器更新。

原配置为 batch 16 + 确定性训练；加速配置为 batch 256 + cuDNN 自动调优 + TF32。以下是训练步吞吐，**不含磁盘解压、文本编码、验证与检查点保存，也不代表完整 epoch 的提速倍率**。

| 模型 | 原配置 样本/秒 | 加速配置 样本/秒 | 吞吐倍率 | 加速配置 PyTorch 峰值已分配显存 |
| --- | ---: | ---: | ---: | ---: |
| TCN | 808 | 13,491 | 16.7× | 354 MiB |
| LSTM | 1,455 | 36,942 | 25.4× | 228 MiB |
| GRU | 1,877 | 37,262 | 19.9× | 224 MiB |
| GRU-D | 54 | 1,012 | 18.6× | 141 MiB |

显存统计不包含驱动、CUDA 上下文、其他进程和完整多模态模型。GRU-D 逐时间步计算衰减与门控，仍比 cuDNN LSTM/GRU 慢。短测速会受 GPU 温度、桌面占用和内核缓存影响。

原始 JSON 位于本机 `runs/benchmarks/5060_before.json` 和 `runs/benchmarks/5060_after.json`（运行产物不纳入 Git）。可使用下面的脚本在当前代码上比较批量和运行模式，脚本只处理合成输入：

```powershell
conda run -n MCP python scripts/benchmark_training.py --batches 16 --output runs/benchmarks/baseline.json
conda run -n MCP python scripts/benchmark_training.py --fast --batches 64 128 256 --output runs/benchmarks/fast.json
```

启动体征神经网络训练：

```powershell
conda run -n MCP python main.py --config configs/vital_thresholds.yaml --models tcn lstm gru grud
```

当前体征 YAML 的模型列表为用户已有的 `[news2]`；上面的 `--models` 显式选择神经网络。本次仅运行短测速和微型回归测试，没有启动完整队列训练。

验证累计 173 项测试通过，包括四种神经网络的真实 CUDA 混合精度训练、CPU/GPU 标准化一致性、TF32 开关与文本缓存隔离、不同训练/验证批量、最大 F1 阈值保存及模型重载。分组运行覆盖现有测试和新增性能回归；原配置测试改为显式传入五模型覆盖列表，不依赖用户当前选中的 `[news2]`。`git diff --check` 通过。

实现依据：[PyTorch CUDA/TF32 文档](https://docs.pytorch.org/docs/stable/notes/cuda.html)及 [PyTorch 性能调优指南](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)。TF32 使用 PyTorch 2.9 之后的新精度接口，较早版本使用兼容接口。
