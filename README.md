# 三模态临床风险预测

当前默认入口为 **30 分钟滑动步长的完整自动实验**。历史窗口仍为 120 分钟、每分钟一个观测点；不是将输入长度改成 30 分钟。默认按未来 **6、3、2、1 小时**依次运行；每个时距自动完成所有候选模型的患者交叉验证、选择一个获选模型、全 Train 重训、Validation 选阈值和 Test 评价，再进入下一时距。所有中间结果保留。

完整实验设计、患者分工、采样对照、硬件测量和局限见 [30 分钟患者交叉验证说明](docs/patient_cv_30min.md)。

## 环境和数据

```powershell
Set-Location D:/development/code/python/MultimodalClinicalPrediction_V2
conda activate MCP
python main.py --help
```

依赖见 `requirements.txt`，Python >=3.10。本机为 RTX 5060 Ti 8GB、内存 32GB；默认自动选择 CUDA/BF16，保留确定性设置。仅体征训练不需要 MacBERT。

本机循环网络层间 dropout 会触发原生退出异常，新实验显式设置 `rnn_dropout: 0.0`，保留投影层 dropout。该兼容设置及验证过程见实验说明；不修改历史配置的默认行为。

当前数据包为 `data/Mutimodel_data/固定训练数据_30min/`。原 `固定训练数据/` 保留用于复核。新包已按预测时间、逐次住院重采样，全量核对保留张量与原值一致；标签不参与选点，患者 train/validation/test 归属不变。

| 集合 | 30 分钟窗口 | 6h 阳性窗口 | 6h 阳性患者 |
|---|---:|---:|---:|
| Train | 282653 | 862 | 98 |
| Validation | 105254 | 288 | 33 |
| Test | 101796 | 235 | 32 |

总计 489703 个窗口、6327 名患者、6717 次住院。以上是元数据计数，训练还会过滤无效输入。相邻 120 分钟输入仍重叠约 75%，因此必须按患者划分和计算置信区间。

数据包包含 `dataset_schema.json`、`sample_index.csv.gz`、`patient_splits.csv`、`splits/`、`tensors/`，以及检验和文本文件。新索引的 `source_sample_id` 对应原包行号；不得单独替换 tensors 后沿用旧索引。数据定义见 [数据契约](docs/data_contract.md)。

## 一次启动完整实验

```powershell
# 审计 Train 三折以及完整 Validation 阈值分工，不训练
python main.py --audit-threshold-splits

# 只以 GRU 为候选，依次完成四个时距的 CV、重训、Validation 和 Test
python main.py --models gru

# 仅运行一个时距；--horizon 覆盖配置中的 horizons 列表
python main.py --models gru --horizon 6

# 默认完整流程：四个时距，每个时距四个候选模型，自动选择并最终评价一个
python main.py

# 只做交叉验证，暂不最终化；用于继续比较采样方案
python main.py --cv-only
python main.py --config configs/patient_cv_30min_uniform.yaml --models gru --cv-only
```

默认配置为 `configs/patient_cv_30min.yaml`。患者分层三折只使用原 Train；每折重新拟合标准化。训练采用患者和事件前时段均衡采样，每批约 6.25% 阳性，每个阳性窗口每轮最多重复 16 次。batch 1024，每轮 32 步，两个 CPU worker 预取未标准化 memmap。采样曝光审计保存在各折目录中。

1 小时任务的阳性窗口较少，配置通过 `positive_fraction_by_horizon` 将其每批阳性比例单独设为 3.125%（32 个），每轮阳性抽样为 1024 次，保持总步数和重复上限不变。其余时距仍为每批 64 个阳性。

早停使用 Recall >=0.80 时的最低 FPR，并报告 0.85/0.90 工作点和整体 AP。原 Validation/Test 不用于 CV 模型评价。默认在每个时距的所有候选完成后，依次按平均 FPR 最低、折间标准差最低、平均 AP 最高选择模型；完全并列时按模型名确定。规则及候选列表在训练前保存，选择记录在读取 Test 前保存。只比较本次启动的候选，不混入历史实验。

```powershell
# 以下仅用于 --cv-only 的手动工作流；默认流程不需要额外执行
# 比较匹配的 CV 方案；输出文件必须不存在
python compare_cv.py --runs-dir runs/patient_cv_30min --output runs/patient_cv_30min/comparison.json

# 选定一个获选方案后，固定 CV 配置并进行最终化
python main.py --finalize-cv "runs/patient_cv_30min/cv/gru/<时间戳>/cv_summary.json"
```

最终化按各折最佳 epoch 中位数在全部 Train 重训，关闭 Platt，完整 Validation 选择满足 Recall >=0.80 时 FP 最小的阈值。保存模型和阈值后才评价 Test。阈值固定，不能根据 Test 再调整；报告患者聚类 bootstrap 95% 区间。同一 CV 运行的最终化会写入锁定记录，阻止意外重复查看 Test。

`--finalize-cv` 沿用已冻结的模型、时距、采样、设备等配置，只允许覆盖输出目录。更改方案需重跑 CV。不要最终化多个候选后根据 Test 挑模型。

## 配置和历史流程

| 配置 | 用途 | 验证分工 |
|---|---|---|
| `patient_cv_30min.yaml` | 默认，仅体征四种神经网络，均衡采样 | Train 内三折；完整 Validation 选阈值 |
| `patient_cv_30min_uniform.yaml` | 均匀采样对照，比较前需匹配预算 | 同上 |
| `train.yaml` | 原多模态流程，含 NEWS2 | 原 holdout 流程 |
| `vital_thresholds.yaml` | 原体征阈值流程；当前工作配置选择 NEWS2，可显式覆盖模型 | 早停、Platt、阈值验证患者拆分 |

以上配置均指向新 30 分钟包。XGBoost 的模型设置保持原实现：120×20 展平为 2400 维、CPU hist 外存训练、验证 AP 早停，原校准和阈值规则。需要运行时显式选择旧入口：

```powershell
python main.py --config configs/vital_thresholds.yaml --models xgboost
python main.py --config configs/train.yaml
```

新患者 CV 入口暂不接受 XGBoost、NEWS2 或多模态模型。统计特征、检验增强、难负样本和更长历史留待后续独立实验；没有将它们混入本次修改。

原多模态流程使用本地 `models/chinese-macbert-base/`，默认不联网下载。TCN、LSTM、GRU、GRU-D 的体征输入均为原 14 通道加 6 个实测标记。GRU-D 使用六个物理量及缺失衰减。NEWS2 为独立删减版规则基线，其规则和限制见 [NEWS2 说明](docs/news2.md)。

原 holdout 的校准和制品说明见 [预测阈值说明](docs/thresholds.md)。旧实验结果属于原始 10 分钟队列，不应直接与新的 30 分钟结果作同口径性能比较。

## 硬件测量和日志

```powershell
python benchmark_training.py --models gru --batches 512 1024 2048 --workers 0 2 --output runs/new_benchmark.json
```

测量仅使用 Train，临时模型丢弃；输出稳态吞吐和 CUDA 显存峰值。该工具不选择科学超参数。批量增大并不总能提高吞吐，GRU-D 的逐步循环仍有明显开销。默认配置以实测吞吐和抽样预算为依据，不保证所有模型持续 100% GPU 利用率。

每次自动流程新建 `runs/patient_cv_30min/experiments/patient_cv/<批次时间戳>/`。批次内按 `h6/`、`h3/`、`h2/`、`h1/` 分开保存：

- `cv/<模型>/<时间戳>/fold_*/`：所有候选的 best/last 权重、每轮历史、预处理参数、采样曝光、折内评价及 `selection_predictions.csv`。未获选模型也保留。
- `model_selection.json`：完整候选排名、选择规则和获选摘要校验和。
- `final/<获选模型>/<时间戳>/`：全量 Train 重训结果、Validation/Test 报告与预测、固定阈值、bootstrap 区间和推理 artifact。

批次根目录的 `experiment_plan.json` 保存预先固定的计划，`experiment_state.json` 保存当前阶段、全部 CV 路径和最终结果路径，`training.log` 汇集全过程日志。任一阶段失败会停止后续工作并保留已有文件，不会从部分候选中继续选模型，也不会清理或覆盖历史结果。重新启动创建新批次，不自动恢复失败批次。`--cv-only` 仍使用原来的 `cv/<模型>/<时间戳>/` 路径。

```powershell
python extract_results.py
```

`extract_results.py` 默认扫描 `runs/patient_cv_30min/`，可用 `--runs-dir` 指定其中某个批次，汇总保存到 `runs/results/`。`training_results.csv` 和 `training_results.json` 包含验证与测试指标、已保存的阈值及患者级 Bootstrap 区间。发现交叉验证运行时，还会生成 `cv_results.csv` 和 `cv_results.json`，包含模型、时距、目标 Recall、各折汇总 FPR 与平均 AUPRC。CSV 中未保存或不可计算的指标留空。使用 `compare_cv.py` 可按匹配的实验协议比较候选。[训练结果字段字典](docs/自动threshold的字段释义.md)

## 推理和验证

```python
from clinical_multimodal.inference.predictor import Predictor
predictor = Predictor("runs/patient_cv_30min/experiments/patient_cv/<批次时间戳>/h6/final/<获选模型>/<时间戳>/artifact", device="cpu")
results = predictor.predict("data/Mutimodel_data/固定训练数据_30min", rows=[0, 1])
```

外部脚本可通过 `python -m pip install -e .` 导入；直接执行 `main.py` 无需安装项目。制品保存输入契约、标准化、固定阈值及校验和；推理不重新选阈值，无可用体征时返回空风险。

```powershell
python -B -m pytest -q -p no:cacheprovider
```

测试覆盖患者隔离、折内预处理、标签盲重采样、重复上限、CV 禁止使用原 Validation/Test、最终阈值冻结、患者 bootstrap、制品加载及原模型行为。完整研究性能需要另行运行正式实验；微型测试和硬件短测不代表临床性能。窗口召回与事件检出率不同，当前没有完整监测暴露数据，不能将窗口 FP 换算成每患者每天报警次数。
