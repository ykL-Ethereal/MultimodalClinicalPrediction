# `training_results.csv` 字段字典

本文覆盖 `runs/threshold_research/results/training_results.csv` 的 163 个字段，也适用于由同一脚本导出的其他运行汇总。每行代表一次模型运行。样本数量、分类计数和比例按预测窗口统计。同一患者或同一次事件可能对应多个窗口。CSV 中的空单元格表示原始结果未提供该值、该字段不适用或该指标不可计算。数值 `0` 是已记录的结果。矩阵和列表以 JSON 文本保存。[结果提取脚本](../extract_results.py)

## 正例与预测输出

- **真实标签 `1`** 表示从预测时点起，未来 `horizon_hours` 小时内发生了上游数据包定义的已确认目标事件。标签 `0` 表示该窗口在该时距下为阴性。本文件各行的 `horizon_hours` 均为 `6`，读取的是 `label_6h`。训练直接使用对应列的 `0/1` 标签。[项目标签说明](../项目描述.md#63-标签与输出) · [标签读取代码](../src/clinical_multimodal/data/fixed_adapter.py#L78)
- **模型输出 `prediction=1`** 表示模型把该窗口判为目标事件阳性并触发报警。可训练模型以 `risk >= threshold` 判阳性，`risk` 使用阈值指定的原始或校准后概率。有效输入低于阈值时输出 `0`，无效输入的预测和报警均为 `null`。NEWS2 单独使用 `score >= alarm_threshold` 输出 `alarm`。[模型推理](../src/clinical_multimodal/inference/predictor.py#L112) · [阈值规则](../src/clinical_multimodal/evaluation/thresholds.py#L23) · [NEWS2 推理](../src/clinical_multimodal/inference/news2.py#L82)
- “病情恶化”可以作为研究语境下的简称，具体口径是**数据包定义的目标事件**。项目资料提到 ICU 相关和死亡相关事件标记，具体纳入标准及时间边界仍需上游标签定义确认。[项目目标说明](../项目描述.md#L7)

## 运行与训练字段

| 字段 | 含义 |
| --- | --- |
| `model` | 模型或基线名称。`gru`、`grud`、`lstm`、`tcn` 是神经网络的体征编码器，`xgboost` 是树模型，`news2` 是规则评分基线。 |
| `mode` | 运行模式。`vital_only` 为仅体征，`multimodal` 为多模态配置。NEWS2 这一行虽标为 `multimodal`，评分实际只读取规则需要的体征。 |
| `run_id` | 运行目录名，通常为北京时间时间戳 `YYYYMMDD_HHMMSS_ffffff`，末段为微秒。 |
| `status` | 运行状态。`completed` 为流程正常完成，`running` 为记录仍处于运行状态，`failed` 为捕获到运行错误，`unknown` 为提取时未读到状态。强制中断后可能留有 `running`。 |
| `horizon_hours` | 运行配置中的预测时距，单位小时。决定读取 `label_1h`、`label_2h`、`label_3h` 或 `label_6h`。 |
| `seed` | 该次运行保存的随机种子，缺少摘要时从训练配置读取。固定队列的原始患者划分有单独的种子。 |
| `device` | 摘要记录的运行设备，如 `cuda:0` 或 `cpu`。无摘要值时回退到配置，因此可能出现 `auto`。 |
| `precision` | 摘要记录的计算精度，如 `bfloat16`、`float16` 或 `float32`。无摘要值时回退到配置，因此可能出现 `auto`。 |
| `selection_metric` | 训练时选择最佳权重或树轮次的指标。神经网络通常为 `auprc`，XGBoost 为 `auprc_ap`，两者均用 Average Precision。 |
| `best_epoch` | 神经网络选中的最佳 epoch，从 `1` 开始。优先读运行摘要，缺失时从历史中的最佳选择分数推导。 |
| `best_iteration` | XGBoost 选中的最佳提升迭代索引，从 `0` 开始。优先读运行摘要，缺失时从历史中的最佳验证 AP 推导。 |
| `best_round` | XGBoost 从 `1` 开始的最佳轮次，等于 `best_iteration + 1`。 |
| `trained_epochs` | 神经网络训练历史中记录的 epoch 总数，包含最佳 epoch 之后为早停判断而运行的轮次。 |
| `trained_rounds` | XGBoost 训练历史中记录的验证 `auprc_ap` 次数，包含最佳轮次之后为早停判断而运行的轮次。 |
| `best_selection_score` | 训练历史里最高的有效选择分数。来自训练期早停验证，当前训练用原始模型输出计算。最终报告可能使用校准后概率，因此列值可能不同。 |
| `best_train_loss` | 最高有效选择分数对应的神经网络 epoch 的平均训练损失。训练使用二元交叉熵，并可对正类加权。该值随最佳验证分数选出。 |
| `parameters` | 神经网络模型的参数元素总数。NEWS2 无拟合参数，记录为 `0`。当前 XGBoost 摘要未提供此值。 |
| `model_version` | 模型制品身份标识，用于追溯该次结果对应的模型或规则版本。 |
| `training_seconds` | 训练阶段耗时，单位秒。神经网络计时覆盖训练循环和早停验证，XGBoost 还包括构建训练及早停矩阵。NEWS2 无训练阶段。 |
| `total_seconds` | 从运行摘要读取的准备、训练与评价阶段计时，单位秒。NEWS2 读取评估阶段计时，包含其数据准备。 |

[运行摘要提取](../extract_results.py#L57) · [神经网络训练](../src/clinical_multimodal/pipelines/train.py) · [XGBoost 训练](../src/clinical_multimodal/pipelines/xgboost.py) · [NEWS2 评估](../src/clinical_multimodal/pipelines/news2.py)

## 验证、测试与阈值选择指标

下列五个前缀各与同一组 21 个后缀组合，构成 105 列。例如 `test_recall` 与 `test_fixed_0_5_recall` 分别是同一测试集在保存阈值和 0.5 下的召回率。

| 前缀 | 数据与阈值口径 |
| --- | --- |
| `validation_` | 最终模型在早停及最佳轮次选择所用的验证子集上的报告。分类指标使用已保存的阈值。NEWS2 使用原始验证集和规则阈值。 |
| `validation_fixed_0_5_` | 同一验证样本和同一概率空间下，以概率 `>= 0.5` 判阳性的对照报告。 |
| `test_` | 最终模型在独立测试集上的报告。分类指标使用测试前保存的阈值。NEWS2 使用规则阈值。 |
| `test_fixed_0_5_` | 同一测试样本和同一概率空间下，以概率 `>= 0.5` 判阳性的对照报告。 |
| `threshold_selection_` | 已保存策略中的阈值来源集合统计。自动策略在独立的阈值验证子集上优化阈值。手工固定策略在早停验证子集记录描述性统计。当前策略只保存样本计数、阈值和分类指标，`status`、`auprc`、`auroc`、`brier`、`log_loss` 后缀对应列留空。 |

| 后缀 | 含义和计算口径 |
| --- | --- |
| `status` | `ok` 表示有两类有效标签，`single_class` 表示只有一类，`no_valid_labels` 表示没有纳入评价的窗口。此状态来自指标报告。 |
| `n` | 有效评价窗口数。要求标签有效且模型输入可用。通常等于 `positive + negative`。 |
| `positive` | 纳入评价且真实标签为 `1` 的窗口数，等于 `TP + FN`（存在分类计数时）。 |
| `negative` | 纳入评价且真实标签为 `0` 的窗口数，等于 `TN + FP`（存在分类计数时）。 |
| `auprc` | 精确率–召回率的排序指标。实现采用 Average Precision（AP），以召回率增量对精确率加权。使用概率或 NEWS2 原始分数的排序，无需分类阈值。 |
| `auroc` | ROC 曲线下面积，衡量正例的概率或 NEWS2 原始分数排在负例之前的能力，无需分类阈值。 |
| `brier` | 概率与真实 `0/1` 标签的均方误差 `mean((p - y)²)`，越小越好。 |
| `log_loss` | 二分类对数损失 `-mean(y × ln(p) + (1-y) × ln(1-p))`，越小越好。实现先将概率截到 `[1e-12, 1 - 1e-12]`。 |
| `threshold` | 该指标组实际应用的阈值。可训练模型用 `[0, 1]` 内的概率阈值，NEWS2 用原始分数阈值。 |
| `sensitivity` | 敏感度 `TP / (TP + FN)`，表示真实阳性窗口中触发报警的比例，与已填报的 `recall` 同口径。没有真实阳性时为空。 |
| `specificity` | 特异度 `TN / (TN + FP)`，表示真实阴性窗口中未报警的比例。没有真实阴性时为空。 |
| `ppv` | 阳性预测值 `TP / (TP + FP)`，表示报警窗口中真实阳性的比例。没有任何报警时为空。 |
| `tp` | 真阳性窗口数。真实标签为 `1`，预测为 `1`。 |
| `tn` | 真阴性窗口数。真实标签为 `0`，预测为 `0`。 |
| `fp` | 假阳性窗口数。真实标签为 `0`，预测为 `1`，即误报。 |
| `fn` | 假阴性窗口数。真实标签为 `1`，预测为 `0`，即漏报。 |
| `precision` | 精确率 `TP / (TP + FP)`。可训练模型在有有效窗口但没有任何报警时记录 `0`，因此此时与空白的 `ppv` 不同。 |
| `recall` | 召回率 `TP / (TP + FN)`，与敏感度相同。分母按窗口统计。 |
| `f1` | 精确率和召回率的调和平均，代码等价地计算 `2 × TP / (2 × TP + FP + FN)`。分母为零时，有有效窗口则记录 `0`，无有效窗口则为空。 |
| `confusion_matrix` | JSON 形式的二阶混淆矩阵 `[[TN, FP], [FN, TP]]`。行是真实 `0, 1`，列是预测 `0, 1`。 |
| `confusion_matrix_order` | 矩阵位置说明文本 `[[TN, FP], [FN, TP]]`。 |

同一集合的主阈值与 `fixed_0_5` 对照共用样本和概率，所以只改变阈值相关统计。AUPRC 与 AUROC 需要两类标签。NEWS2 使用未校准的原始规则分数，当前报告提供排序指标和部分分类指标，概率损失、`fixed_0_5` 对照及部分扩展分类字段留空。提取器直接保留空值。[二分类指标](../src/clinical_multimodal/evaluation/metrics.py) · [阈值分类指标](../src/clinical_multimodal/evaluation/thresholds.py#L32) · [NEWS2 指标](../src/clinical_multimodal/pipelines/news2.py#L17)

## 验证和测试报告的附加字段

下列 9 个后缀各有 `validation_` 和 `test_` 两列，共 18 列。

| 后缀 | 含义 |
| --- | --- |
| `calibrated` | 报告是否使用校准后的概率。可训练模型在启用校准时使用 Platt，NEWS2 为 `False`。 |
| `invalid_inputs` | 因输入不可用而被模型判为无效的窗口数。评估的 `n` 还要求标签有效，因此标签缺失和输入无效是两个概念。 |
| `prediction_seconds` | 该集合的评价阶段计时，单位秒。计时覆盖预测、指标计算和逐窗口预测文件写入。 |
| `metric_unit` | 指标统计单位。可训练模型记录 `window`，即逐窗口。 |
| `validation_role` | 验证报告用途。`early_stopping_selection` 表示用于训练早停与最佳轮次选择的子集，`test_validation_role` 留空。 |
| `continuous_warning_status` | 连续预警评价状态。`not_computed` 表示尚未计算。NEWS2 使用 `not_computed_requires_event_and_exposure_contract`。 |
| `continuous_warning_missing_fields` | 连续预警需要但缺少的字段列表，当前可训练模型记录 `event_time`、`monitor_end`、`monitor_start`。 |
| `continuous_warning_reason` | 连续预警指标缺失原因。可靠的事件时间和完整监测暴露区间是计算事件检出率、预警提前量及每患者日假报警数的前提。 |
| `positive_rate` | 评价窗口阳性比例，等于对应 `positive / n`，以 `0–1` 的小数保存。`n` 为零或缺失时留空。 |

[报告字段构造](../src/clinical_multimodal/pipelines/threshold.py#L101) · [阳性比例提取](../extract_results.py#L165)

## 保存的阈值策略及来源

策略优先来自运行目录的 `threshold.json`，其次为运行摘要、验证或测试报告中的策略，以及制品目录的 `threshold.json`。NEWS2 的阈值见 `validation_threshold` 和 `test_threshold`，以下概率策略字段留空。[阈值来源提取](../extract_results.py#L94)

| 字段 | 含义 |
| --- | --- |
| `threshold` | 保存的分类概率阈值，与对应模型的验证、测试主指标共用。 |
| `threshold_strategy` | `recall_constraint` 在满足目标召回率的候选阈值中最大化精确率。`max_f1` 最大化 F1。`fixed` 直接使用配置提供的数值。 |
| `target_recall` | `recall_constraint` 对阈值选择数据设定的最低窗口召回率，如 `0.8` 即 80%。其他策略留空。测试集召回率由 `test_recall` 给出。 |
| `threshold_probability_space` | 阈值适用的概率空间。`raw` 为模型 logit 经 sigmoid 后的概率，`calibrated` 为 Platt 校准后的概率。 |
| `threshold_horizon_hours` | 保存阈值所适用的预测时距，单位小时。加载制品时会与推理时距核对。 |
| `threshold_classification_rule` | 判阳规则 `p >= threshold`，等于阈值也判阳性。 |
| `threshold_unit` | 阈值应用的单位。`window` 表示对每个预测窗口独立判断。 |
| `threshold_tie_break` | 自动选择时的并列规则。`higher_threshold` 表示优化目标相同则取较高阈值。固定策略也保存该值，执行时直接用给定阈值。 |
| `threshold_source_split` | 阈值统计所在的原始患者划分，当前为 `validation`。 |
| `threshold_source_role` | `threshold_validation` 表示独立阈值验证子集，`manual_fixed` 表示固定阈值并在早停验证子集上记录统计。 |
| `threshold_source_independent_of_model_and_calibrator_fit` | `True` 表示阈值统计所用患者与模型训练、早停选择和校准所用患者互斥。手动固定策略为 `False`。 |
| `threshold_source_selection_metrics_are_optimization_set_statistics` | 报告把 `threshold_selection_*` 标记为阈值来源集合的统计。当前代码对手动固定策略也保存 `True`，是否实际搜索阈值应看 `threshold_strategy`。 |
| `threshold_source_total_windows` | 阈值来源集合筛选前的窗口总数。 |
| `threshold_source_excluded_windows` | 来源集合中因标签或输入无效被排除的窗口数。`total_windows - excluded_windows` 等于 `threshold_selection_n`。 |

[验证患者分区与阈值策略](thresholds.md#阈值在哪一步固定) · [策略字段构造](../src/clinical_multimodal/pipelines/threshold.py#L72)

## 制品路径和错误信息

| 字段 | 含义 |
| --- | --- |
| `artifact_exists` | 提取时 `artifact_dir` 是否存在且为目录。这个字段只检查目录。 |
| `artifact_dir` | 模型制品目录的绝对路径。即使目录缺失也会记录预期路径。 |
| `run_dir` | 本次运行目录的绝对路径，通常包含配置、状态、训练历史和评估报告。 |
| `error_type` | 运行失败时 `status.json` 保存的异常类名。 |
| `error` | 运行失败时 `status.json` 保存的异常消息。详细堆栈可查运行目录的 `training.log`。 |
| `extraction_issues` | 提取过程中发现的问题，如 JSON 无法读取、已完成运行缺报告或缺制品目录。多条问题以分号连接。 |

[运行日志](../src/clinical_multimodal/run_logging.py) · [结果提取](../extract_results.py#L118)

## 用当前 GRU 行核对口径

`test_n=303524`，其中 `test_positive=693`，阳性比例约为 `0.2283%`。使用保存的校准概率阈值 `test_threshold≈0.0005612`，`test_tp=588`，`test_fn=105`，所以窗口召回率为 `588 / 693 ≈ 84.85%`。同一行的 `test_fixed_0_5_tp=0`，说明在同一批窗口上把阈值设为 `0.5` 时，没有真实阳性窗口触发报警。阈值选择集的 `threshold_selection_recall≈81.68%` 来自另外一批窗口（`threshold_selection_n=78155`）。[当前结果 CSV](../runs/threshold_research/results/training_results.csv)
