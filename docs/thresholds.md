# 可复现的窗口级预测阈值

## 实际代码审计

检查对象是修改前的工作区实际代码，而非仅依据背景描述。

- `FixedCohortBundle` 读取固定包的 `patient_splits.csv`、索引和 split 数组；检查同一 `patient_group_id` 的所有窗口只能在一个 split，同一次住院只能对应一个患者。训练专属标准化已存在。
- 实际数据为 1,459,630 个窗口、6,327 名患者，包装划分为 60%/20%/20% 患者，划分种子 20261015。输入长度 120 分钟；体征专用预处理输出 14 个包装特征和 6 个真实观测标记，共 20 通道。GRU-D 按原实现从中使用六个物理量及缺失信息。
- 神经网络 TCN/LSTM/GRU/GRU-D 和 XGBoost 均已实现。原 `configs/train.yaml` 默认是多模态、关闭校准，模型列表不含 XGBoost；体征 CLI 默认四种神经网络。新增的 `configs/vital_thresholds.yaml` 才明确启用五种体征模型和 Platt 校准。
- 1/2/3/6 小时标签存在，每次运行训练一个时距。新制品逐模型、逐时距独立保存阈值，加载时核对时距，禁止套用其他时距阈值。
- 没有发现交叉验证或 OOF 训练入口。神经网络按验证 AUPRC 选最佳 epoch，恢复最佳权重；XGBoost 按验证 AP 早停，推理限制为最佳轮次。没有将 train+validation 合并重训的步骤。
- 原可选 Platt 校准将验证患者拆成早停选择与校准两部分；没有自动阈值选择或 `best_f1_thr` 实现。`training.threshold` 原先为 null，显式设置后才计算阈值指标。背景中的测试集 `best_f1_thr≈0.195414` 只能是历史描述，未写入任何新配置或制品。

## 阈值在哪一步固定

保留原始 train/validation/test 患者划分，仅拆 validation 患者。固定种子、排序后分组洗牌，按**当前预测时距是否存在阳性窗口**尽量分层：

1. `threshold_fraction=0.30` 验证患者用于阈值选择。
2. 剩余验证患者中，`calibration_fraction=0.50` 用于 Platt 拟合，即原验证患者约 35%。
3. 其余约 35% 验证患者只用于早停/最佳权重选择。

关闭校准时，早停/阈值约为 70%/30%。手动 fixed 不需要为优化预留阈值集；保存的指标来自早停集，仅作该固定阈值的描述，并明确标为非独立优化验证。极小分层只有一个患者时保持患者隔离、退回确定性不分层切分；不保证所有角色都有两类。自动阈值集、Platt 集或训练/早停有效窗口缺少类别时直接失败，绝不借用测试患者或静默复用其他角色。没有患者窗口跨隔离集合。

流程为训练及最佳权重恢复 → 独立校准患者拟合 Platt → 独立阈值患者预测并应用已拟合校准器 → 选择并保存阈值 → 固定阈值评价测试集。阈值选择后不重训、不重拟合校准器。没有 OOF 及其校准交叉拟合依赖。

阈值集未参与模型权重/早停/校准器拟合，但它的指标仍是**阈值优化集统计量**，不是无偏最终验证结果。`validation_report.json` 来自早停集，也不应解释为独立最终验证结果。独立 `test_report.json` 才用于最终评价，测试后不得据此重设目标或阈值。重复根据测试结果挑选模型也会破坏该独立性。

## 三种策略与边界

配置 `training.threshold_strategy`：

- `recall_constraint`：在窗口 recall ≥ `target_recall` 的候选中选 precision 最大者，并列取更高阈值。默认 0.80 只是可配置研究目标，不是已验证临床标准。
- `max_f1`：F1 最大，并列同样取更高阈值。
- `fixed`：必须显式填写 `training.threshold`；不优化。为防配置歧义，自动策略的 threshold 必须为 null。

在不同预测概率的精确边界上扫描，排序复杂度 O(n log n)，不会用粗网格漏掉边界或拆开相同概率的样本。候选按降序扫描，目标值相同选择第一个；全程规则为 `p >= threshold`。阈值允许 [0,1] 端点，处理恰好为 0 或 1 的概率。没有预测阳性时 precision 定义为 0；无正标签时 recall 不可定义，自动优化拒绝该输入。

启用 Platt 时只在 calibrated 概率上选阈值，关闭时只用 raw 概率。输出概率在训练导出和推理中均使用同一 NumPy sigmoid/Platt 路径。不同设备/批量/浮点精度可能仍产生模型概率的微小差异，边界附近的分类也可能受影响；正式复现实验应固定设备、精度、依赖版本及批量。

## 保存与推理

运行根目录及 `artifact/threshold.json` 保存 threshold、threshold_strategy、适用的 target_recall、probability_space、horizon_hours、分类规则、选择数据 precision/recall/F1/specificity/混淆矩阵/有效与排除窗口数。来源记录包括验证角色、患者 ID、样本 ID、原始患者划分文件 SHA256、确定性分区 SHA256、有效选择样本 ID SHA256、种子与分配比例。制品 manifest 包含完整策略快照，`threshold.json` 纳入文件完整性校验。

推理只执行原始模型输出 → 已保存校准器 → 已保存阈值，输出 `risk`、整数 `prediction`、布尔 `alarm` 和概率空间。无体征输入返回 null 风险和 null 分类；禁止把无效输入解释为阴性。缺少阈值策略、时距不符、概率空间不符均报错；不向旧制品静默补 0.5，也不从历史测试结果补阈值。原产物文件保留，但缺少完整策略的旧制品需要按新流程重新训练后使用分类推理。

`evaluation/thresholds.py` 中的应用函数被分类指标、预测导出和推理共用。epoch 历史只保留 raw 概率排序及概率指标，不把 calibrated 阈值应用到 raw 概率。

## 测试与事件指标

`test_report.json.metrics` 报固定阈值下 precision/recall/F1/specificity、TP/TN/FP/FN、`[[TN,FP],[FN,TP]]` 混淆矩阵，同时保留 AUROC/AUPRC、Brier、log loss 和可靠性分箱。`fixed_0_5` 是同一概率空间的 0.5 对照。当前不计算测试最优阈值，不保留测试后重新选择入口。

当前主策略是窗口级 recall，**不等于患者检出率或事件检出率**。实际索引包含患者、住院、预测时间、time_to_event_minutes 及上下界；没有直接 event_time，也没有监测开始/结束或完整有效暴露区间。TTE 来自区间中点，不能凭此假定精确事件时刻；预测窗口起止也不能充当监测暴露时长。因此尚不输出事件检出率、首次报警提前量、每患者每天假报警次数，更不能用窗口 FPR 代替假报警/患者日。报告列出缺失字段。

已有 `evaluation/events.py` 接受调用者明确提供的事件/暴露契约，但当前训练流程没有连接该契约。阈值策略明确保存逐窗口报警、无合并、冷却 0 分钟；尚未引入任何报警合并/冷却优化。未来若接入完整事件及监测数据，须先在验证数据确定并保存规则，再用于测试。

## 直接运行（项目根目录，已安装依赖）

```powershell
# 只审计患者分区及元数据，不训练、不选择数值阈值
python main.py --config configs/vital_thresholds.yaml --horizon 6 --audit-threshold-splits

# 训练 + Platt + 独立验证阈值选择 + 固定阈值测试，自动新建运行目录
python main.py --config configs/vital_thresholds.yaml --horizon 6 --models xgboost lstm tcn gru grud

# 四个时距分别运行，阈值互相独立（这是完整队列实验，可能耗时较长）
foreach ($h in 1, 2, 3, 6) {
    python main.py --config configs/vital_thresholds.yaml --horizon $h --models xgboost lstm tcn gru grud
}

# 只评价某个已保存制品；替换下列 artifact 路径，不会寻找新阈值
python main.py --config configs/vital_thresholds.yaml --artifact runs/threshold_research/vital_only/gru/<时间戳>/artifact --evaluation-output runs/threshold_research/retest_new.json --device cpu

# 阈值单元测试与五种模型微型端到端测试
python -B -m pytest -q -p no:cacheprovider tests/test_thresholds.py
python -B -m pytest -q -p no:cacheprovider
```

若默认 python 不在项目环境，请使用 `D:/development/software/anaconda3/envs/MCP/python.exe` 替代。配置默认写入全新 `runs/threshold_research/`，每次仍自动创建时间戳目录，已有制品不覆盖。评估命令拒绝已存在输出路径。

如需 F1 对照，将研究 YAML 另存后设 `threshold_strategy: max_f1`；手动策略设 `threshold_strategy: fixed`、`threshold: 0.5`。两者不改变测试隔离规则。自动选择有意与训练绑在同一次运行内，避免对已经训练、早停使用了全部验证患者的旧模型事后声称拥有独立阈值验证集。

## 局限

固定一次患者划分可审计、可复现，但不估计抽样不确定性。约 30% 验证患者上的正例尤其在 1 小时任务较少；窗口高度相关，不可把窗口数当作独立患者样本量。研究目标在此样本上达到不保证新患者/测试集也达到。当前实现没有患者 bootstrap 置信区间、跨机构验证或临床报警效果验证。完整队列的阈值和性能必须实际重新训练得到，微型测试结果不能作为性能结论。

## 本次实际检查结果（2026-11-02）

使用 MCP 环境 Python 运行全套测试：`164 passed in 73.78s`，包含五种模型的微型训练、校准、独立阈值选择、保存加载与推理，以及原有 CPU/CUDA 流程。初次 XGBoost 重载在 Windows 沙箱内因路径规范化访问被拒绝；获准在沙箱外重跑通过。`git diff --check` 通过（仅现有 LF/CRLF 提示）。没有运行 146 万窗口的完整五模型×四时距训练，没有新真实队列阈值或性能结果，没有覆盖旧训练产物。

已对真实包逐时距运行 `--audit-threshold-splits`，审计产物曾生成于 `runs/threshold_research/audits/`，随后按用户要求清理，可运行上述命令重新生成。下表完全来自元数据审计，**未检查所有张量的体征可用性**，最终有效窗口数可能更少，不是模型评价结果。

| 时距/小时 | 早停患者 | 校准患者 | 阈值患者 | 阈值窗口 | 阈值阳性窗口 |
|---|---:|---:|---:|---:|---:|
| 1 | 443 | 443 | 380 | 77,411 | 25 |
| 2 | 442 | 444 | 380 | 80,629 | 72 |
| 3 | 443 | 444 | 379 | 132,050 | 114 |
| 6 | 443 | 443 | 380 | 78,155 | 191 |

各时距按其自身阳性标签分层，角色患者及窗口数量不必相同。患者标识、预测时间已由读取契约检查完整；事件时间与监测起止字段缺失，因此事件/暴露指标未完成。后续仍需实际完整训练、固定阈值测试及阈值稳定性评估。

## 本次修改文件清单

- 新增 `src/clinical_multimodal/evaluation/thresholds.py`：优化、并列规则、统一阈值应用及策略验证。
- 新增 `src/clinical_multimodal/pipelines/threshold.py`：独立患者划分、来源审计、阈值拟合及报告补充。
- 修改 `src/clinical_multimodal/config.py`、`src/clinical_multimodal/runner.py`：策略参数验证与训练/审计/固定阈值测试 CLI。
- 修改 `src/clinical_multimodal/pipelines/train.py`、`src/clinical_multimodal/pipelines/xgboost.py`、`src/clinical_multimodal/pipelines/evaluate.py`：训练和独立评价接入。
- 修改 `src/clinical_multimodal/evaluation/calibration.py`、`src/clinical_multimodal/evaluation/metrics.py`、`src/clinical_multimodal/evaluation/predictions.py`：Platt 精度一致性、分类指标和预测导出。
- 修改 `src/clinical_multimodal/training/checkpoint.py`、`src/clinical_multimodal/training/trainer.py`、`src/clinical_multimodal/inference/predictor.py`：策略制品及完整性校验、epoch 指标概率空间、加载固定决策。
- 新增 `tests/test_thresholds.py`；修改 `tests/fixed_fixture.py`、`tests/helpers.py`、`tests/test_pipeline.py`、`tests/test_vital_only.py`：覆盖新行为并适配旧微型夹具。
- 新增 `configs/vital_thresholds.yaml`、本文 `docs/thresholds.md`；修改 `README.md`：研究入口、审计结论、命令和限制。

工作区原有的其他未提交改动予以保留，上表不把它们算作本次新增工作。
