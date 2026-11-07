# 当前架构

唯一入口为根目录 `main.py`，默认读取 `configs/train.yaml`。`runner.main` 依次选择 `training.models`，`train_pipeline` 为每个模型创建北京时间戳目录，记录实际配置和日志，再执行数据准备、训练、评价和制品导出。

固定数据路径为 `data/Mutimodel_data/固定训练数据`，仅使用 `FixedCohortBundle` 与 `FixedPreparedDataset`。分片 NPZ 按需加载，最多缓存两个分片；分片内随机批处理避免全量张量常驻内存。原始表格、旧 tensor 格式及演示数据路径已移除。

数值参数只在训练患者上拟合。体征输入含 14 个包装特征及 6 个实测 mask。TCN/LSTM/GRU 共用 20 通道输入；GRU-D 从中选择六个物理量和真实观测 mask，计算窗口内分钟间隔。各编码器输出末时刻和均值的投影表示。

其余分支共用：检验快照残差 MLP、冻结 MacBERT、块与文书两级注意力、年龄性别静态 MLP、concat/gated 融合。训练与推理均由 `model.vital_encoder` 选择编码器，制品包含该字段并校验全部文件。

GRU-D 为单层论文式衰减单元，LSTM/GRU 层数可配置。时间 mask 只表示右侧填充，特征缺失由实测 mask 表示。标准化训练均值为零，GRU-D 首步缺失按均值初始化，不使用验证/测试统计量，也不把前向填充当作实测。

每个模型、每次运行的日志和制品隔离。共享冻结文本缓存通过正文和骨干身份生成键，可跨模型安全复用。完整队列的标准化仍在每次模型运行时拟合；当前未实现跨运行标准化缓存或断点恢复。

## 体征单模态与 XGBoost

`model.vital_only` 或 `--vital_only` 选择独立路径。读取器仅解压体征、观测标记、预测时间及需要的标签；`VitalNumericPreprocessor` 仅拟合训练患者体征统计；数据集与 collate 跳过文书；`VitalRiskModel` 只包含现有体征编码器和线性分类头。CPU 与 CUDA 均保持 `[B,120,20]` 输入，GRU-D 的内部物理量选择不变。

`pipelines/xgboost.py` 使用同一单模态预处理，分批把有效训练/选择验证样本展平到磁盘 `.npy` 文件，DataIter 将这些文件供给 CPU `ExtMemQuantileDMatrix`，验证矩阵使用训练矩阵的量化切点。使用 binary:logistic、hist 和验证集 average precision 自定义指标早停；预测通过最佳 iteration 范围取得 margin，再执行 sigmoid 或独立患者 Platt 校准。不使用神经网络 Trainer、优化器或 AMP；不读取检验、病历及静态信息。

单模态输出路径为 `runs/vital_only/<model>/<时间>/`，训练后自动写验证和测试预测 CSV。格式版本 2 的制品记录模式、20 通道布局和校验和；神经网络保存 weights.pt，XGBoost 保存 model.ubj，均不导出文本骨干。Predictor 自动按制品重建路径；原多模态格式版本 1 继续兼容。NEWS2 仍走独立规则流程。
