import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def binary_metrics(labels, probabilities, threshold=None, n_bins=10):
    y, p = np.asarray(labels), np.asarray(probabilities, dtype=float)
    if y.shape != p.shape or y.ndim != 1 or not np.isin(y, [0, 1]).all():
        raise ValueError("Metrics require matching vectors of binary labels and probabilities")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid probabilities")
    if threshold is not None and not 0 < threshold < 1:
        raise ValueError("Invalid evaluation threshold")
    result = {"n": len(y), "positive": int((y == 1).sum()), "negative": int((y == 0).sum()),
              "auprc": None, "auroc": None, "brier": None, "log_loss": None,
              "calibration_curve": [], "threshold": threshold, "sensitivity": None,
              "specificity": None, "ppv": None, "status": "no_valid_labels"}
    if len(y) == 0:
        return result
    both = len(np.unique(y)) == 2
    result.update(status="ok" if both else "single_class", brier=float(np.mean((p-y)**2)))
    bounded = np.clip(p, 1e-12, 1 - 1e-12)
    result["log_loss"] = float(-np.mean(y * np.log(bounded) + (1-y) * np.log1p(-bounded)))
    if both:
        result.update(auprc=float(average_precision_score(y, p)), auroc=float(roc_auc_score(y, p)))
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    for i in range(n_bins):
        selected = bins == i
        if selected.any():
            result["calibration_curve"].append({"bin": i, "count": int(selected.sum()),
                                               "mean_probability": float(p[selected].mean()),
                                               "event_rate": float(y[selected].mean())})
    if threshold is not None:
        pred = p >= threshold
        tp, tn = int(np.sum(pred & (y == 1))), int(np.sum(~pred & (y == 0)))
        result.update(tp=tp, tn=tn, fp=int(np.sum(pred & (y == 0))), fn=int(np.sum(~pred & (y == 1))))
        result["sensitivity"] = tp / result["positive"] if result["positive"] else None
        result["specificity"] = tn / result["negative"] if result["negative"] else None
        result["ppv"] = tp / int(pred.sum()) if pred.any() else None
    return result

