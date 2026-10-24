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
