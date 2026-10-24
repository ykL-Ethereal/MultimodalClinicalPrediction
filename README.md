# 三模态临床风险预测

项目只读取本项目 `data/Mutimodel_data/固定训练数据/` 中已经包装好的固定队列，将生命体征、检验快照、中文病历及年龄性别用于未来事件预测。支持 1、2、3、6 小时标签，每次实验选择一个窗口。默认预测未来 6 小时。

## 启动

在安装依赖的 Python 环境中，从项目根目录执行：

```bash
python main.py
```

本机项目配置的环境为 `MultiCP`。系统默认的 Python 3.12 尚未安装 PyTorch，应先激活环境：

```bash
conda activate MultiCP
python main.py
```

新环境可通过 `python -m pip install -r requirements.txt` 安装依赖，要求 Python 3.10 或以上。无需安装本项目为命令行工具。首次使用默认 MacBERT 时需要下载 `hfl/chinese-macbert-base`；也可在配置中指定本地模型目录并开启 `local_files_only`。

默认依次训练 **TCN、LSTM、GRU**。修改 [configs/train.yaml](configs/train.yaml) 中的 `training.models` 可选择模型或加入 GRU-D：

```yaml
training:
  models: [tcn, lstm, gru, grud]
```

也可以临时选择模型：

```bash
python main.py --models lstm gru
python main.py --models grud
python main.py --validate-only
python main.py --config configs/train.yaml
```

`--validate-only` 检查所有分片及病历文本，可能需要较长时间，不会训练或下载语言模型。默认启动会进行完整正式训练；本次代码验证没有启动 146 万样本的完整训练。

## 唯一数据来源

```text
data/Mutimodel_data/固定训练数据/
  dataset_schema.json
  sample_index.csv.gz
  patient_splits.csv
  lab_feature_catalog.csv
  note_text.jsonl
  note_index.npz
  splits/
  tensors/shard_*.npz
```

`configs/train.yaml` 的数据路径相对于配置文件解析。项目不再提供旧 tensor 包适配、原始表格接入、合成数据生成或演示训练入口。测试中的微型固定格式夹具仅用于自动验证。

当前数据包含 1,459,630 个窗口、6,327 名患者、6,717 次住院。每个窗口包含过去 120 分钟的生命体征及 593 项检验快照。程序保留包装好的患者划分与事件标签，只在训练患者上拟合数值标准化；不会重新生成标签或重新划分患者。

## 模型

| 配置名称 | 生命体征分支 | 实现与输入 |
|---|---|---|
| `tcn` | 单向因果膨胀残差卷积 | 14 个包装特征 + 6 个实测标记，共 20 通道 |
| `lstm` | PyTorch `nn.LSTM` | 单向，默认两层，支持序列右侧填充 |
| `gru` | PyTorch `nn.GRU` | 单向，默认两层，输入与 LSTM/TCN 相同 |
| `grud` | GRU-D 缺失感知循环单元 | 六个物理量、实测掩码、窗口内距上次实测的分钟间隔；单层 |

TCN、LSTM、GRU 直接读取同一套 20 通道输入。GRU-D 选取心率、脉率、呼吸频率、血氧、收缩压、舒张压，使用输入衰减及隐藏状态衰减，并将实测 mask 加入门控。填充值不能被当作实测值。标准化后的训练均值为零；窗口开始前的实测历史不可恢复，因此初始上次观测值采用训练均值，间隔从窗口起点计算。这一边界约定随实现固定。

LSTM/GRU 复用 PyTorch 官方实现。GRU-D 依据原论文式 (10)–(16)，仅实现官方 GRU 不具备的衰减单元；其候选状态遵循论文的 reset-before-matrix 公式。详见 [docs/references.md](docs/references.md)。

四种模型都将末个有效时刻和有效时刻均值拼接，再映射到相同维度。检验残差 MLP、冻结 MacBERT + 块/文书两级注意力、静态 MLP、concat/gated 融合及损失函数在模型间共用。`rnn_hidden` 控制循环隐藏维度，`rnn_layers` 仅控制 LSTM/GRU 层数。相同隐藏维度不表示参数量相同，运行摘要记录总参数量。

## 日志和结果

每次训练新建目录，重复启动不会覆盖旧结果：

```text
runs/
  tcn/<北京时间戳>/
  lstm/<北京时间戳>/
  gru/<北京时间戳>/
  grud/<北京时间戳>/
```

每个目录独立保存：

- `training.log`：准备、训练进度、标准输出与错误堆栈；默认每 100 个批次及每个 epoch 输出进度。
- `config.json`、`status.json`：实际配置及 running/completed/failed 状态。
- `history.json`：逐 epoch 损失和验证指标。
- `best.pt`、`last.pt`：最佳参数及最新参数/优化器状态。
- `validation_report.json`、`test_report.json`、`run_summary.json`：评价及性能摘要。
- `data_report.json`、`preprocessing_preview.json`：数据检查和训练集标准化参数。
- `artifact/`：包含模型种类、配置、权重、输入契约、标准化参数、文本骨干和校验和的推理制品。

文本块向量缓存位于 `runs/macbert_cache/`，多模型可复用；它是长期可复用缓存。故障会写入该模型日志和状态文件并终止此次启动，已有结果保留。下次启动创建新的运行目录，不支持自动恢复中断训练。

## 评价与推理

验证集 AUPRC 用于选择 epoch，测试集用于最终评价。可配置独立患者概率校准。缺失动态模态的样本返回空风险。相关接口保留为 Python API：

```python
from clinical_multimodal.data.fixed_adapter import FixedCohortBundle
from clinical_multimodal.inference.predictor import Predictor

bundle = FixedCohortBundle('data/Mutimodel_data/固定训练数据', require_targets=False)
predictor = Predictor('runs/gru/<运行时间>/artifact')
results = predictor.predict(bundle, rows=[0, 1], batch_size=2)
```

从外部脚本调用上述 API 时，可安装本项目（`python -m pip install -e .`）或将 `src` 加入 `PYTHONPATH`。直接运行 `main.py` 不需要这一步。

## 验证与代码位置

```bash
python -B -m pytest -q -p no:cacheprovider
```

测试覆盖固定包契约、患者泄漏、训练集标准化、四种编码器的因果性和填充隔离、缺失模态、梯度、GRU-D 公式与时间间隔、四种模型训练及制品重新加载、分模型日志和独立进程预测。测试夹具和微型本地语言模型均在测试结束后删除。

- `main.py`、`src/clinical_multimodal/runner.py`：启动和模型选择。
- `config.py`：配置及检查。
- `data/fixed_adapter.py`、`data/fixed_dataset.py`：唯一数据读取与按分片批处理。
- `preprocessing/fixed_numeric.py`：训练集标准化；`text.py`、`text_cache.py`：MacBERT 及缓存。
- `models/vital_tcn.py`、`vital_rnn.py`、`vital_grud.py`：生命体征编码器。
- `pipelines/train.py`、`training/`、`run_logging.py`：训练、制品和日志。
- `evaluation/`、`inference/`：评价与预测。

本项目用于研究实验，性能结论需要完整队列实验支持。
