# NEWS2 规则基线

依据用户提供的 Haj Husin 等（2025）《Predicting COVID-19 Patient Disposition Using the National Early Warning Score 2: A Retrospective Cohort Study》，DOI [10.7759/cureus.79610](https://doi.org/10.7759/cureus.79610)，并按用户随后提供的 NEWS2 代码更新区间和缺测归一化规则。报警阈值保留 PDF 第 6 页表 2 的原始总分 ≥2。该研究评价既有 NEWS2，并非提出新的 NEWS2 评分系统。

## 数据适配

当前配置仅使用 HR、RR、SpO2、SBP 四项。体温、意识状态、补充氧气在数据契约中没有对应字段，因此不参与评分，不添加占位分数。数据包的脉率和舒张压也不进入当前四参数评分。

- 心率项仅选择 `hr`；HR 缺测时跳过该项，不以 `pr` 替代。
- SpO2 采用图 1 的 **Scale 1**。当前数据缺少选择 Scale 2 的适应证与补氧状态，不推断 Scale 2。
- 每项取预测时点之前、120 分钟窗口中最后一次有效观测；不计算窗口均值或窗口最高分。各项可来自不同分钟。
- 高频字段只在对应 `*_mask == 0` 时视为实际观测，忽略掩码下的填充值。收缩压使用 `sbp_last_mask == 1` 标识可用的历史最近值，允许该值来源于窗口开始之前；不把 `vital_observed_mask` 误作血压历史可用标记。不会新增血压时效阈值。
- 按用户代码，缺测项既不计分，也不计入理论最高分；不以正常值或零分插补。只要至少一项有观测，就返回部分评分，并用 `partial_input` 状态标明不完整输入。全部缺测时返回 `raw_score: null`、`normalized_score: null`、`max_score: 0`、`alarm: null`。这一缺测处理取代先前的整样本排除约定。
- 不读取病历、检验、静态信息、未来事件或标签作为评分输入，不拟合标准化或模型参数。

结果明确标记为 `variant: reduced`。默认配置总分范围为 0–12，不能称为完整 NEWS2。论文的 COVID-19 急诊住院结局与本项目未来 1/2/3/6 小时事件标签不同；沿用 ≥2 是可复现的参考设置，不代表该阈值已经在本项目或删减版评分中验证。

`raw_score` 为可用项得分之和，`max_score` 为这些可用项的配置满分之和，`normalized_score = raw_score / max_score`。各项理论最高分由配置中的 `max_score` 明确列出，默认均为 3，不能用区间数量代替；可用但得 0 分的正常观测仍计入满分。归一化结果处于 [0,1]，不是风险概率。输出 `score` 保留为 `raw_score` 的兼容字段。

例如 `HR=135, RR=22, SpO2=90, SBP=105` 对应 `3+2+3+1=9` 分，满分 12，归一化为 **0.75**；同一例子缺测 HR 时为 `6/9`，而非 `6/12`。

## 独立规则配置

所有分档、分值、每项理论最高分、字段映射、缺测规则、区间端点约定、归一化方式和报警阈值均放在 [configs/news2_rules.yaml](../configs/news2_rules.yaml)。代码只解释配置，没有内嵌临床阈值。

图 1 的整数分档对应如下；表中为整数边界：

| 参数 | 3 分 | 2 分 | 1 分 | 0 分 |
|---|---|---|---|---|
| 心率（次/分） | ≤40 或 ≥131 | 111–130 | 41–50 或 91–110 | 51–90 |
| 呼吸频率（次/分） | ≤8 或 ≥25 | 21–24 | 9–11 | 12–20 |
| SpO2 Scale 1（%） | ≤91 | 92–93 | 94–95 | ≥96 |
| 收缩压（mmHg） | ≤90 或 ≥220 | 91–100 | 101–110 | 111–219 |

当前 v2 配置使用 `interval_closed: upper`，严格对应用户代码的 `value <= upper`：`cutpoints` 为包含端点的上限，最后一档隐含上限正无穷，无需将非有限数写入 YAML。例如 RR 的 `[8, 11, 20, 24]` 表示 `(-∞,8]`、`(8,11]`、`(11,20]`、`(20,24]`、`(24,+∞)`；RR=20.9 得 2 分。`scores` 长度必须等于断点数加一，不进行隐式四舍五入。

论文正文称氧饱和度计分为 0–2，与图 1 的 Scale 1（0–3）不一致；表 1 部分端点还使用了 `<`/`>`。当前分值采用用户代码中与图 1 一致的 Scale 1 分值，连续值边界以用户代码为准。报警比较仍对原始总分使用 ≥2，不将该阈值套用到归一化分数。

先前 v1 制品的规则快照仍按其原有下界包含区间和 `require_all` 缺测策略加载；新运行使用 v2 配置。旧制品的评分含义不会被当前规则文件改变。

`configs/train.yaml` 的 `news2.rules_file` 路径相对于该配置文件解析；可另存规则文件后修改路径。加载时检查规则字段、断点顺序、分档数、分值类型、阈值范围及来源字段。评分源字段不匹配时会报错，不自动将缺少字段当作正常。

## 运行与输出

```bash
python main.py --models news2
python main.py --models tcn lstm gru grud news2
```

`training.models` 也支持 `news2`。NEWS2 直接使用固定患者验证/测试划分，按 `data.horizon` 评价，不启动训练或下载语言模型。`training.threshold`、概率校准、优化器及设备设置不作用于 NEWS2；NEWS2 的报警阈值使用独立规则文件。

每次运行在 `runs/news2/<北京时间戳>/` 保存：

- `config.json`、`status.json`、`training.log`、`run_summary.json`：配置、运行状态与摘要。
- `validation_report.json`、`test_report.json`：`metrics` 保存原始分数的 AUROC、AUPRC（average precision）及阈值下混淆矩阵、敏感度、特异度和 PPV；`normalized_metrics` 另外保存归一化分数的 AUROC 和 AUPRC，不臆设归一化报警阈值。保留样本总数、无效样本数、部分观测样本数和各项缺测数。
- `validation_predictions.jsonl`、`test_predictions.jsonl`：逐样本 `raw_score`、`max_score`、`normalized_score`、报警、各项原始值与得分、来源通道、时间索引、缺测及删减项；按批流式写入，不保留整个队列的诊断对象。
- `artifact/news2_rules.yaml`、`artifact/manifest.json`：实际使用的规则快照、输入契约、预测窗口、规则版本与完整性校验。加载制品后无需原规则文件。

规则总分和归一化分数均不是概率，输出 `risk: null`；Brier、log loss 和概率校准曲线不适用，报告中留空。无有效样本或只有一个类别时，无法计算的指标同样留空。

```python
from clinical_multimodal.data.fixed_adapter import FixedCohortBundle
from clinical_multimodal.inference.news2 import NEWS2Predictor
from clinical_multimodal.pipelines.evaluate import evaluate_pipeline

bundle = FixedCohortBundle('data/Mutimodel_data/固定训练数据', require_targets=False)
predictor = NEWS2Predictor('runs/news2/<运行时间>/artifact')
predictions = predictor.predict(bundle, rows=[0, 1], batch_size=256)
# 大批量推理可遍历 predictor.iter_predict(...)，避免收集全部预测。

# 重新评价时使用有标签的数据包；evaluate_pipeline 自动识别 NEWS2 制品。
report = evaluate_pipeline('runs/news2/<运行时间>/artifact',
                          'data/Mutimodel_data/固定训练数据', 'runs/news2_evaluation.json')
```

测试使用微型固定包核对整数与小数边界、包含式上限、全部 16 种四参数缺测组合、用户的 0.75 示例、HR 缺测不回退 PR、历史收缩压、配置满分、旧版规则兼容、报警等号、制品加载、命令行入口和原始/归一化评价结果。
