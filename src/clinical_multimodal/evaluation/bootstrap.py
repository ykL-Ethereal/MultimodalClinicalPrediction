"""Patient-cluster percentile intervals for a previously frozen decision rule."""
import numpy as np


def patient_bootstrap(labels, probabilities, patients, threshold, *, replicates=1000, seed=42):
    y, p, ids = np.asarray(labels), np.asarray(probabilities), np.asarray(patients)
    if y.shape != p.shape or y.shape != ids.shape or not len(y):
        raise ValueError("Bootstrap requires aligned, nonempty patient predictions")
    unique, inverse = np.unique(ids, return_inverse=True)
    pred = p >= threshold
    groups = np.zeros((len(unique), 4), np.int64)
    columns = np.where(y == 1, np.where(pred, 0, 3), np.where(pred, 2, 1))  # TP,TN,FP,FN
    np.add.at(groups, (inverse, columns), 1)
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in ("recall", "precision", "fpr", "fp")}
    for _ in range(replicates):
        tp, tn, fp, fn = groups[rng.integers(len(unique), size=len(unique))].sum(axis=0)
        for key, numerator, denominator in (("recall", tp, tp + fn), ("precision", tp, tp + fp),
                                             ("fpr", fp, fp + tn), ("fp", fp, 1)):
            if denominator:
                draws[key].append(float(numerator / denominator))
    return {"method": "patient_cluster_percentile", "confidence": 0.95, "replicates": replicates,
            "patients": len(unique), "seed": seed, "threshold_reselected": False,
            "conditional_on_fixed_model_and_threshold": True,
            "intervals": {key: {"lower": float(np.quantile(v, .025)) if v else None,
                                  "upper": float(np.quantile(v, .975)) if v else None,
                                  "valid_replicates": len(v)} for key, v in draws.items()}}
