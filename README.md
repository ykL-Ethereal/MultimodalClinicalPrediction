# 三模态临床风险预测

体征专用的可复现阈值实验请使用 `configs/vital_thresholds.yaml`：五种模型、120×20 输入、Platt 校准、独立验证患者阈值选择。完整流程、实际代码审计、训练/阈值选择/测试命令及局限见 [预测阈值说明](docs/thresholds.md)。旧制品缺少完整阈值策略时，新的分类推理会明确报错；不会补入测试集最优阈值。

项目只读取本项目 `data/Mutimodel_data/固定训练数据/` 中已经包装好的固定队列，将生命体征、检验快照、中文病历及年龄性别用于未来事件预测。支持 1、2、3、6 小时标签，每次实验选择一个窗口。默认预测未来 6 小时。

## 启动

以下命令均从项目根目录执行。Windows PowerShell 使用本机已有的 `MCP` 环境：

```powershell
Set-Location D:/development/code/python/MultimodalClinicalPrediction
conda activate MCP
python -m pip install -r requirements.txt
python main.py --help
```

要求 Python 3.10 或以上；没有 `MCP` 环境时，可先用 `conda create -n MCP python=3.11` 创建再激活。已安装依赖可跳过安装步骤。未使用 Conda 时，在自己的 Python 环境安装相同依赖即可；无需将本项目安装为命令行工具。

启动前确认固定数据包位于下文“唯一数据来源”列出的目录。默认多模态配置从 `models/chinese-macbert-base/` 加载本地 MacBERT，且 `text.local_files_only: true`，不会自动下载；该目录需包含模型权重、配置及 tokenizer 文件。需要在线加载时，将 `configs/train.yaml` 的 `text.model_name` 改为 `hfl/chinese-macbert-base`，并设 `text.local_files_only: false`。仅体征训练不需要 MacBERT。

先检查数据，再启动默认训练：

```powershell
python main.py --validate-only
python main.py
```

默认依次训练 **TCN、LSTM、GRU、GRU-D**，并评价 **删减版 NEWS2**。修改 [configs/train.yaml](configs/train.yaml) 中的 `training.models` 可选择模型：

```yaml
training:
  models: [tcn, lstm, gru, grud, news2]
```

也可以临时选择模型：

```bash
python main.py --models lstm gru
python main.py --models grud
python main.py --models news2
python main.py --validate-only
python main.py --config configs/train.yaml
```

`--validate-only` 检查所有分片，多模态模式还检查病历文本，并拟合训练集数值预处理统计，可能需要较长时间；不会训练或下载语言模型。默认启动会进行完整队列训练，模型按列表顺序依次运行。

两份配置对应不同实验入口：

| 配置 | 输入与模型 | 校准和阈值 | 默认运行根目录 |
|---|---|---|---|
| `configs/train.yaml` | 多模态 TCN/LSTM/GRU/GRU-D，加 NEWS2 基线 | 神经网络不校准，按默认 recall_constraint 在独立验证患者上选择 raw 概率阈值；NEWS2 使用规则阈值 | `runs/` |
| `configs/vital_thresholds.yaml` | 仅体征 XGBoost/LSTM/TCN/GRU/GRU-D | Platt 校准，独立验证患者选择 calibrated 概率阈值，目标窗口 recall 为 0.80 | `runs/threshold_research/` |

常用训练参数如下；未提供的参数使用所选 YAML 及代码默认值：

| 参数 | 用法 |
|---|---|
| `--config` | 选择 YAML；默认 `configs/train.yaml` |
| `--models` | 覆盖模型列表，例如 `--models lstm gru` |
| `--vital-only` / `--vital_only` | 切换仅体征输入 |
| `--horizon` | 选择 1、2、3、6 小时之一；默认 6，每次运行仅训练一个时距 |
| `--output-dir` | 指定运行根目录，例如 `--output-dir runs/my_experiment`；相对路径按当前工作目录解析 |
| `--device` | `auto`、`cpu` 或 `cuda`，覆盖训练/评价设备；XGBoost 保持 CPU |
| `--validate-only` | 只做数据准备和检查 |
| `--audit-threshold-splits` | 只审计验证患者分区和元数据，不训练、不选择数值阈值 |

训练轮数、早停和批量等在 YAML 的 `training` 中修改：`epochs` 是最大轮数，`patience` 是早停等待轮数，`batch_size` 是神经网络批量，`learning_rate` 是学习率，`seed` 控制随机种子。XGBoost 使用独立 `xgboost` 节的 `num_boost_round`、`early_stopping_rounds`、`batch_size` 等参数。CLI 没有 `--epochs`、`--batch-size` 或 `--threshold` 参数。

## 仅用体征训练

`--vital_only`（也接受 `--vital-only`）使用原有 14 个体征通道及 6 个真实观测标记，共 `[N,120,20]`。不加入检验、病历或年龄性别，不加载 MacBERT，不拟合检验/静态统计量。原始数据及 mask 方向保持不变。

```bash
python main.py --vital_only
python main.py --vital_only --models tcn lstm gru grud
python main.py --vital_only --models xgboost
python main.py --vital_only --models tcn lstm gru grud xgboost
python main.py --vital_only --validate-only
```

也可设置 `model.vital_only: true`。使用 CLI 的 `--vital-only` / `--vital_only` 且未指定 `--models` 时，固定运行 TCN/LSTM/GRU/GRU-D；仅在 YAML 中设置 `model.vital_only: true` 时，仍按该 YAML 的 `training.models` 运行。显式 `--models` 优先。NEWS2 仍为独立规则流程，显式选择时不经过单模态分类头。

单模态神经网络使用“体征编码器 → 线性分类头 → 风险概率”，不创建多模态融合层。GRU-D 继续使用真实观测标记及原有衰减逻辑。全部体征不可用时返回 `risk: null`、`status: invalid_input_no_vitals`。

XGBoost 仅支持体征单模态，固定将相同预处理后的 120×20 窗口按时间、通道顺序展平为 2400 维，无 `--xgb_features` 参数。零值保持普通数值（可能是标准化均值或 mask 位），不设置为 XGBoost 的缺失哨兵。标准化只用训练集统计。参数位于 `xgboost` 配置节；采用 CPU `hist` 和 `ExtMemQuantileDMatrix` 外存训练，独立于神经网络的 CUDA/混合精度设置。

全部 1,459,630 个窗口的 float32 展平特征约 13.05 GiB，训练还有额外内存和磁盘开销。代码只为训练及选择验证集分批保存有效样本的 `.npy` 特征，默认每批 16384 行，并由 XGBoost 分批构建外存页；测试集直接分批预测。运行目录中的 `feature_cache/` 不属于推理制品，可在不需要审计特征时自行清理。需要 `xgboost>=3.1,<4`；未安装时普通神经网络仍可运行，选择 XGBoost 会提示安装依赖。

## 训练并选择预测阈值

需要仅体征、Platt 校准及独立阈值选择时，可直接按以下顺序执行：

```powershell
# 检查完整数据包，仅体征模式不读取病历文本
python main.py --config configs/vital_thresholds.yaml --validate-only

# 审计当前时距的早停、校准和阈值患者分区
python main.py --config configs/vital_thresholds.yaml --horizon 6 --audit-threshold-splits

# 先运行一个模型：训练 → 校准 → 阈值选择 → 验证/测试评价
python main.py --config configs/vital_thresholds.yaml --horizon 6 --models gru

# 或按配置运行全部五种体征模型
python main.py --config configs/vital_thresholds.yaml --horizon 6

# 训练结束后提取本次实验目录下的结果
python extract_results.py --runs-dir runs/threshold_research --output-dir runs/threshold_research/results
```

上述单模型和五模型命令是两种启动选择，每次执行都会新建运行目录。研究配置已经设置仅体征模式，无需额外加 `--vital-only`；额外加该参数而不指定 `--models` 会改为四种神经网络，跳过 XGBoost。

每个时距需要独立训练、校准和选择阈值。完整四时距实验可在 PowerShell 执行：

```powershell
foreach ($h in 1, 2, 3, 6) {
    python main.py --config configs/vital_thresholds.yaml --horizon $h
    if ($LASTEXITCODE -ne 0) { break }
}
```

阈值策略在 YAML 的 `training` 中设置：默认 `threshold_strategy: recall_constraint`，在窗口 recall ≥ `target_recall: 0.80` 的候选中选 precision 最大者；`max_f1` 选择最大 F1；`fixed` 必须同时设置数值 `threshold`，例如 0.5。自动策略需保留 `threshold: null`。目标 recall 约束作用于阈值选择集，不保证测试集达到相同召回率。

训练使用原训练患者；验证患者进一步分为早停、校准、阈值选择三种角色。阈值固定后才评价测试集。`validation_report.json` 来自早停验证集，`threshold_selection_*` 是阈值优化集统计，`test_report.json` 用于最终测试评价。患者分区、策略细节和事件级指标限制见 [预测阈值说明](docs/thresholds.md)。

## GPU 加速

默认配置自动选择 CUDA，当前已在 `MCP` 环境的 RTX 5060 Ti 上验证。终端可执行：

```powershell
conda run -n MCP python main.py
```

`text.device` 和 `training.device` 均为 `auto`，无 CUDA 时回退 CPU；指定 `cuda:0` 可要求必须使用指定显卡。训练、验证、推理及 MacBERT 均使用所选设备。CUDA 路径还在显卡上完成训练集统计拟合、批次标准化、缺失值掩码、时间特征和模态质量计算；通过锁页内存及非阻塞传输减少数据搬运等待，使用融合 AdamW 和 `inference_mode`。

配置中的 `precision: auto` 在支持时使用 BF16，否则使用 FP16，并为 FP16 训练启用梯度缩放。设为 `float32` 可关闭混合精度。文本向量缓存包含实际设备类型和精度，切换精度或 CPU/GPU 后会生成独立缓存。`training.deterministic: false` 启用 cuDNN 自动调优；需要确定性实验时改为 `true`。

磁盘解压、文本分词、JSON 报告和 scikit-learn 评价/校准保留在 CPU。NEWS2 保持独立的 NumPy 规则基线，无需加载 PyTorch 或文本模型。上述加速不改变患者划分、标签或训练专属标准化规则。

接口细节参考 [PyTorch 混合精度示例](https://docs.pytorch.org/docs/stable/notes/amp_examples.html)和 [AdamW 文档](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html)。

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

当前数据包含 1,459,630 个窗口、6,327 名患者、6,717 次住院。每个窗口包含过去 120 分钟的生命体征及 593 项检验快照。程序保留包装好的 train/validation/test 患者划分与事件标签，只在训练患者上拟合数值标准化；校准和阈值流程仅进一步划分原 validation 患者的用途。

## 模型

| 配置名称 | 生命体征分支 | 实现与输入 |
|---|---|---|
| `tcn` | 单向因果膨胀残差卷积 | 14 个包装特征 + 6 个实测标记，共 20 通道 |
| `lstm` | PyTorch `nn.LSTM` | 单向，默认两层，支持序列右侧填充 |
| `gru` | PyTorch `nn.GRU` | 单向，默认两层，输入与 LSTM/TCN 相同 |
| `grud` | GRU-D 缺失感知循环单元 | 六个物理量、实测掩码、窗口内距上次实测的分钟间隔；单层 |
| `news2` | 独立规则评分基线，无需训练 | 使用 HR、RR、SpO2 Scale 1 和 SBP；输出原始分数、可用项满分、归一化分数及报警 |
| `xgboost` | 仅体征单模态的树模型 | 完整 120×20 窗口展平，2400 维；验证 AP 早停 |

NEWS2 参考 Haj Husin 等（2025），并按用户提供代码采用包含上限的区间：仅使用 HR、RR、SpO2、SBP，省略体温、意识状态和补充氧气。规则分档、分值、各项满分、字段映射及报警阈值（默认原始总分 ≥2）单独保存在 [configs/news2_rules.yaml](configs/news2_rules.yaml)，由 `news2.rules_file` 指定。缺测项不计分也不计入满分，归一化分数为可用项总分除以其理论满分；全部缺测时返回空分数和满分 0。原始分数及归一化分数均不是概率，不计算 Brier 或 log loss。完整说明与推理 API 见 [docs/news2.md](docs/news2.md)。

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
  news2/<北京时间戳>/
  vital_only/<tcn或lstm或gru或grud或xgboost>/<北京时间戳>/
  threshold_research/vital_only/<模型>/<北京时间戳>/
```

每个目录独立保存：

- `training.log`：准备、训练进度、标准输出与错误堆栈；默认每 100 个批次及每个 epoch 输出进度。
- `config.json`、`status.json`：实际配置及 running/completed/failed 状态。
- `history.json`：逐 epoch 损失和验证指标。
- `best.pt`、`last.pt`：最佳参数及最新参数/优化器状态。
- `validation_report.json`、`test_report.json`、`run_summary.json`：评价及性能摘要。
- `validation_predictions.csv`、`test_predictions.csv`：最佳模型的逐样本概率、有效性和标签；启用阈值时含报警列。
- `validation_partitions.json`：早停、校准和阈值选择患者/样本分区及校验信息。
- `threshold.json`、`threshold_validation_predictions.csv`：已固定的阈值策略、选择集统计及对应逐样本预测。
- `data_report.json`、`preprocessing_preview.json`：数据检查和训练集标准化参数。
- `artifact/`：包含模型种类、配置、权重、输入契约、标准化参数、文本骨干和校验和的推理制品。

文本块向量缓存位于 `runs/macbert_cache/`，多模型可复用；它是长期可复用缓存。故障会写入该模型日志和状态文件并终止此次启动，已有结果保留。下次启动创建新的运行目录，不支持自动恢复中断训练。

NEWS2 运行保存相同命名的配置、日志、状态与评价摘要，另存逐样本 `*_predictions.jsonl` 和包含规则快照的 `artifact/`；不生成神经网络权重、训练历史或文本缓存。

运行开始时终端会打印输出目录；可查看该目录的 `training.log` 跟踪进度。只有 `status.json` 的 `status` 为 `completed` 才表示该次运行正常完成。`failed` 时检查 `error` 和日志末尾堆栈；`running` 但进程已退出可能表示运行被外部中断。新训练需重新启动，会创建新目录。

## 提取和阅读结果

使用独立脚本自动提取已保存的结果，无需加载模型或重新训练，也不需要额外依赖：

```bash
# 提取全部运行（包括失败、未完成的运行）
python extract_results.py

# 提取体征单模态的最新运行
python extract_results.py --vital-only --latest --models tcn lstm gru grud xgboost

# 自定义输入与输出目录；输入也可以是某次运行目录
python extract_results.py --runs-dir runs/vital_only --output-dir runs/results

# 提取独立验证患者阈值实验的结果
python extract_results.py --runs-dir runs/threshold_research --output-dir runs/threshold_research/results

# 仅提取阈值实验中各模型/时距/种子的最新运行
python extract_results.py --runs-dir runs/threshold_research --output-dir runs/threshold_research/results --vital-only --latest

# 查看完整参数说明
python extract_results.py --help
```

| 提取参数 | 说明 |
|---|---|
| `--runs-dir` | 搜索的运行根目录，也接受单次运行目录；默认项目 `runs/`，递归查找并跳过缓存和制品目录 |
| `--output-dir` | CSV/JSON 输出目录；默认项目 `runs/results/`，不随 `--runs-dir` 改变 |
| `--models` | 仅提取指定模型，例如 `--models gru xgboost` |
| `--vital-only` | 仅保留体征单模态；提取脚本仅接受此连字符拼写 |
| `--latest` | 按模式、模型、时距、种子保留最新时间戳的运行 |

默认输出 `runs/results/training_results.csv`（UTF-8 BOM，可用 Excel 打开）和
`training_results.json`（保留完整验证/测试报告、校准曲线及已保存的阈值策略），终端同时显示主要指标和测试阈值、Precision、Recall、F1。
每次运行一行，包含状态、模型、模式、预测时距、种子、设备、最佳轮次、实际训练轮数、
耗时、验证/测试集样本数与阳性率、AUPRC、AUROC、Brier、log loss、阈值指标及制品路径。
新增字段包括 `validation_` / `test_` 前缀的 precision、recall、f1、confusion_matrix 及其顺序，
`validation_fixed_0_5_` / `test_fixed_0_5_` 前缀的 0.5 阈值对照指标，
以及 threshold、threshold_strategy、target_recall、threshold_probability_space、分类规则和来源信息。
`threshold_selection_` 前缀是阈值选择集的优化统计，与最终验证/测试指标分开；
`*_continuous_warning_*` 保留连续预警指标的计算状态、缺失字段及原因。CSV 内矩阵和列表使用 JSON 文本。
阈值策略依次从运行目录 `threshold.json`、摘要、评价报告或 `artifact/threshold.json` 读取，不从配置推测实际阈值。
缺失指标保留为空，不填零，也不根据混淆矩阵补算；旧报告或 NEWS2 未保存的新增字段同样为空。
`best_epoch` 从 1 开始；XGBoost 的 `best_iteration` 从 0 开始，`best_round` 为对应的第几轮（从 1 开始）。
`best_selection_score` 来自训练历史；最终报告指标来自已保存的最佳模型评估，启用校准时可能与历史分数不同。
`--latest` 按模式、模型、预测时距和种子分别选取最新时间戳，保留其实际状态，不会用旧的成功运行替代最新失败运行。
提取时会覆盖输出目录内同名的两个汇总文件；原始训练结果不受影响。

阅读汇总时先检查 `status`、`error` 和 `extraction_issues`，再比较最终测试指标。常用列按以下方式分组：

| 字段 | 含义 |
|---|---|
| `model`、`mode`、`horizon_hours`、`seed`、`run_id` | 实验身份；不同时距的阈值分别解释 |
| `test_auprc`、`test_auroc`、`test_brier`、`test_log_loss` | 最终测试集的排序与概率指标 |
| `test_threshold`、`test_precision`、`test_recall`、`test_f1`、`test_specificity` | 已固定阈值在测试集上的分类指标 |
| `test_tp`、`test_tn`、`test_fp`、`test_fn`、`test_confusion_matrix` | 测试混淆计数，矩阵顺序见 `test_confusion_matrix_order` |
| `threshold_strategy`、`target_recall`、`threshold_probability_space` | 已保存的阈值策略及适用概率空间 |
| `threshold_selection_*` | 阈值选择集的优化统计 |
| `test_fixed_0_5_*` | 相同概率空间下 0.5 阈值的对照结果 |
| `test_continuous_warning_*` | 连续预警指标状态、缺失字段和原因；`not_computed` 不表示数值为零 |

验证集对应指标使用 `validation_` 前缀。JSON 顶层 `runs` 的每一项包含 `summary`、`reports` 和 `threshold_policy`，便于进一步分析。旧运行没有保存的新指标不会通过重新提取自动生成；没有匹配运行时脚本打印提示，不写输出文件。不同阈值策略若需要同时保留，应分别汇总或不加 `--latest`，因为最新运行分组不包含策略。

## 评价与推理

验证集 AUPRC 用于选择 epoch，测试集用于最终评价。可配置独立患者概率校准。缺失动态模态的样本返回空风险。相关接口保留为 Python API：

只需重新评价已有制品时，可使用以下 CLI；将 `<时间戳>` 替换为实际运行目录名：

```powershell
python main.py --config configs/vital_thresholds.yaml --artifact "runs/threshold_research/vital_only/gru/<时间戳>/artifact" --evaluation-output runs/threshold_research/retest_new.json --device cpu
```

该命令默认评价测试集，沿用制品的模型、校准器、预测时距及已保存阈值；输出路径必须尚不存在。它只生成指定的评价 JSON，独立 JSON 不会自动成为 `extract_results.py` 的运行记录；原训练报告仍从各运行目录提取。

```python
from clinical_multimodal.data.fixed_adapter import FixedCohortBundle
from clinical_multimodal.inference.predictor import Predictor

bundle = FixedCohortBundle('data/Mutimodel_data/固定训练数据', require_targets=False)
predictor = Predictor('runs/gru/<运行时间>/artifact')
results = predictor.predict(bundle, rows=[0, 1], batch_size=2)
```

从外部脚本调用上述 API 时，可安装本项目（`python -m pip install -e .`）或将 `src` 加入 `PYTHONPATH`。直接运行 `main.py` 不需要这一步。

体征单模态及 XGBoost 制品不包含文本骨干，建议直接传入数据包路径，让预测器按制品选择读取模式：

```python
predictor = Predictor('runs/vital_only/gru/<运行时间>/artifact', device='cpu')
results = predictor.predict('data/Mutimodel_data/固定训练数据', rows=[0, 1])
```

路径替换为 `runs/vital_only/xgboost/<运行时间>/artifact` 即可加载树模型。单模态制品使用格式版本 2，含 `input_layout.json`；XGBoost 权重为 `model.ubj`，不生成 `best.pt`/`last.pt`，预测自动限制为最佳轮次。多模态制品使用格式版本 1；无论格式版本，当前加载器均要求完整阈值策略，缺少该策略的旧制品需按新流程重新训练。

## 常见问题

| 现象 | 检查或处理 |
|---|---|
| 导入依赖失败或找不到 PyTorch | 确认已激活安装依赖的环境，可用 `python -c "import sys; print(sys.executable)"` 核对解释器 |
| 找不到固定数据包 | 检查 YAML 的 `data.root` 和 `dataset_schema.json`；YAML 相对路径按配置文件所在目录解析 |
| 本地 MacBERT 加载失败 | 检查 `models/chinese-macbert-base/` 文件或调整 `text.model_name`；仅体征流程不需要文本模型 |
| GPU 显存不足 | 减小 YAML 的 `training.batch_size`；多模态文本编码还可减小 `text.batch_size` |
| 自动阈值或校准提示缺少类别/患者 | 检查当前时距的独立验证分区和有效样本；不要将测试患者加入拟合，详见阈值文档 |
| 提取提示没有匹配运行 | 核对 `--runs-dir`，并检查模型、单模态筛选条件；脚本读取保存的运行目录，不解析训练日志文本 |
| 提取结果含空值或 NOTE | 查看 `status`、`error`、`extraction_issues` 及原报告；可能是旧报告、尚未完成或缺少产物 |

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
