import numpy as np
from ..data.fixed_adapter import FixedCohortBundle
from ..evaluation.metrics import binary_metrics
from ..evaluation.report import save_report
from ..inference.predictor import Predictor


def evaluate_pipeline(artifact, data_root, output, *, split="test", device="cpu", batch_size=16):
    predictor = Predictor(artifact, device=device)
    bundle = FixedCohortBundle(data_root)
    rows = bundle.rows(split)
    predictions = predictor.predict(bundle, rows=rows, batch_size=batch_size)
    y, label_mask = bundle.labels(rows, predictor.config.data.horizon)
    valid = label_mask & np.array([p["risk"] is not None for p in predictions], dtype=bool)
    probability = np.array([p["risk"] if p["risk"] is not None else 0. for p in predictions])
    metrics = binary_metrics(y[valid], probability[valid], predictor.manifest["threshold"])
    return save_report(output, metrics, split=split,
                       extra={"calibrated": predictor.calibrator is not None,
                              "invalid_inputs": sum(p["risk"] is None for p in predictions)})
