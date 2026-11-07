import numpy as np
from ..data.fixed_adapter import FixedCohortBundle
from ..evaluation.metrics import binary_metrics
from ..evaluation.report import save_report
from ..io import read_json
from .threshold import evaluation_details


def evaluate_pipeline(artifact, data_root, output, *, split="test", device="auto", batch_size=16):
    from pathlib import Path
    if read_json(Path(artifact) / "manifest.json").get("model_type") == "news2":
        from ..inference.news2 import NEWS2Predictor
        from .news2 import evaluate_news2
        return evaluate_news2(NEWS2Predictor(artifact), FixedCohortBundle(data_root), output,
                              split=split, batch_size=batch_size)
    from ..inference.predictor import Predictor
    predictor = Predictor(artifact, device=device)
    bundle = FixedCohortBundle(data_root, vital_only=predictor.config.model.vital_only)
    rows = bundle.rows(split)
    predictions = predictor.predict(bundle, rows=rows, batch_size=batch_size)
    y, label_mask = bundle.labels(rows, predictor.config.data.horizon)
    valid = label_mask & np.array([p["risk"] is not None for p in predictions], dtype=bool)
    probability = np.array([p["risk"] if p["risk"] is not None else 0. for p in predictions])
    metrics = binary_metrics(y[valid], probability[valid], predictor.manifest["threshold"])
    return save_report(output, metrics, split=split,
                       extra={"calibrated": predictor.calibrator is not None,
                              "invalid_inputs": sum(p["risk"] is None for p in predictions),
                              **evaluation_details(y[valid], probability[valid], predictor.policy, bundle)})
