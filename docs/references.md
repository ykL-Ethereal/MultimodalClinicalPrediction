# 实现依据

- [PyTorch LSTM](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html)：复用 `nn.LSTM`，设置 `batch_first=True`、单向；多层 dropout 只作用于层间。
- [PyTorch GRU](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html)：复用 `nn.GRU`，单向；官方候选状态的 reset 门顺序与原论文存在差别。
- [Packed sequences](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html)：LSTM/GRU 用于右侧填充；全空序列的占位步骤在输出处屏蔽。
- [Che et al., Recurrent Neural Networks for Multivariate Time Series with Missing Values, Scientific Reports, 2018](https://www.nature.com/articles/s41598-018-24271-9)：GRU-D 输入/隐藏衰减及缺失掩码，按式 (10)–(16) 实现；输入衰减为逐特征对角形式。GRU-D 为六个标准化物理量计算衰减，训练均值在该空间为零。
- [作者团队 GRU-D 参考实现](https://github.com/peterche1990/gru-d)：历史 Keras/TensorFlow 实现说明需要 value/mask/timestamp；本项目采用 PyTorch，未引入整套旧框架依赖，也未复制其代码。
- [Bai et al., TCN, 2018](https://arxiv.org/abs/1803.01271)：因果膨胀残差时间卷积。
- [MacBERT](https://aclanthology.org/2020.findings-emnlp.58/)：冻结中文预训练骨干。
- [Hierarchical Attention Networks](https://aclanthology.org/N16-1174/)：分层聚合的背景；本项目层级为文本块与文书。

LSTM/GRU 直接使用成熟模块。GRU-D 的输入衰减、隐藏衰减和缺失门控不由 `nn.GRU` 提供，故仅这部分使用标准 `nn.Linear` 和 PyTorch 张量运算实现。GRU-D 为单层，聚合方式与其他分支统一为末时刻加均值，其整体模型不等同于论文的原始分类器。模型优劣需要在相同划分下通过实验比较。
