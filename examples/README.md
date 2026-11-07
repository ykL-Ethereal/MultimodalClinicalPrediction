# 固定数据包推理示例

从项目根目录运行，先安装本项目（`python -m pip install -e .`）或将 `src` 加入 `PYTHONPATH`。

```python
from clinical_multimodal.data.fixed_adapter import FixedCohortBundle
from clinical_multimodal.inference.predictor import Predictor

bundle = FixedCohortBundle('data/Mutimodel_data/固定训练数据', require_targets=False)
predictor = Predictor('runs/gru/<运行时间>/artifact', device='cpu')
results = predictor.predict(bundle, rows=bundle.rows('test')[:16], batch_size=16, diagnostics=True)
```

训练由 `python main.py` 启动。制品中的配置决定体征编码器和预测窗口；完整保留 `artifact/text_backbone/` 以支持离线推理。仅支持当前固定数据包格式。

仅体征训练使用 `python main.py --vital_only --models gru`，XGBoost 使用 `python main.py --vital_only --models xgboost`。两者制品无需文本骨干，加载方法相同：

```python
predictor = Predictor('runs/vital_only/xgboost/<运行时间>/artifact')
results = predictor.predict('data/Mutimodel_data/固定训练数据', rows=[0, 1], batch_size=256)
```

预测器按制品自动选择体征读取器，不要求检验目录或病历文件。手工构造数据包时使用 `FixedCohortBundle(root, vital_only=True, require_targets=False)`；无体征可用的行返回空风险。
