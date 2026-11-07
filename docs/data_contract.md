# 固定包装数据契约

唯一格式：`data/Mutimodel_data/固定训练数据` 中的 `dataset_schema.json`、全局样本索引、患者划分、检验目录、文本及偏移索引、划分 ID 和分片 NPZ。

- 标签窗口为 1、2、3、6 小时；生命体征历史为 120 分钟。
- 体征原始顺序为 hr, pr, rr, spo2, hr_mask, pr_mask, rr_mask, spo2_mask, sbp_last, sbp_mins_since_last_norm, sbp_last_mask, dbp_last, dbp_mins_since_last_norm, dbp_last_mask。
- 额外 `vital_observed_mask` 为六个物理量的实际观测标记，追加后形成 20 通道网络输入。
- 高频 `*_mask` 为缺失指示，与 observed mask 方向相反；血压 `*_last_mask` 表示历史值可用，不能代替实际观测 mask。
- 检验按 code/unit 唯一确定，最新值、差值各有独立 mask。模型追加时间与可用标记，共每项六通道。
- 静态信息顺序为 age_years, sex_female, sex_male，另含独立 mask。
- 同一患者只允许一个划分。分片行、全局 sample_id、split ID、预测时间与标签必须一致；有效文书不能晚于预测时点。
- 只使用训练患者的真实观测估计标准化参数；训练观测不足两条的检验被禁用。mask 后的非数值填充不参与模型。

`FixedCohortBundle` 初始化检查元数据和文件覆盖，每个分片首次读取时检查张量与标签/时间一致性。`python main.py --validate-only` 遍历全部分片和文书。训练流程不重新划分患者、不生成标签、不修改数据包。

GRU-D 的间隔单位为分钟，从当前 120 分钟窗口首步起算，初始间隔为零；之后根据前一步是否实际观测递推。窗口外上次实测未知，初始上一观测取标准化训练均值零。与原始不规则记录上运行的 GRU-D 实验需要区分这一输入边界。

## 体征单模态

`FixedCohortBundle(root, vital_only=True)` 只要求体征相关 schema、样本索引、患者划分、split ID 以及分片中的 `vital_features`、`vital_observed_mask`、`prediction_second`；有监督训练还要求标签一致。不要求检验目录、病历正文/索引、检验或静态数组。`--vital_only --validate-only` 只扫描此模式需要的数据。

不迁移或翻转任何 mask：高频缺测位 1 表示当前分钟缺测；血压 last mask 的 1 表示历史值可用；observed mask 的 True 表示真实观测。高频缺测位置可能保留因果前向填充值，不能把全部缺测位的数值清零。检验、静态、文书和标签 mask 的 True 继续表示有效。历史数据 Notebook 中若把血压 last mask 称为“缺失标志”，应以此处和当前读取器校验口径为准。

单模态预处理与多模态的体征部分一致：仅标准化六个物理量，保留原 14 通道顺序，再追加 hr/pr/rr/spo2/sbp/dbp 六个实测标记。训练观测不足两条的物理量按既有规则置零并关闭追加观测标记。血压历史可用性不会被当作真实观测。现有有效窗口判定保持基于可用体征历史。

单模态契约 `fixed-cohort-vital-v1` 不包含检验、静态或文本身份，预处理版本为 `vital-numeric-v1`。XGBoost 使用同一 `[120,20]` 数组的 C-order 展平：第 `t*20+c` 列对应第 t 分钟第 c 通道，共 2400 列。完整名字和顺序保存于制品 `input_layout.json`，不包含样本 ID、患者 ID、时间戳、标签或结局特征。
