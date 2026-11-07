"""Window decisions and reproducible threshold optimization (never called at inference)."""
import numbers
import numpy as np


STRATEGIES = {"recall_constraint", "max_f1", "fixed"}


def validate_threshold(threshold):
    if (isinstance(threshold, (bool, np.bool_)) or not isinstance(threshold, numbers.Real)
            or not np.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError("Missing/invalid threshold: expected a finite number in [0, 1]")
    return float(threshold)


def probabilities_vector(probabilities):
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be a finite vector in [0, 1]")
    return p


def apply_threshold(probabilities, threshold, *, probability_space="raw", expected_space=None):
    """The single inclusive classification rule used by metrics, exports and inference."""
    threshold = validate_threshold(threshold)
    if probability_space not in ("raw", "calibrated") or (
            expected_space is not None and probability_space != expected_space):
        raise ValueError("Threshold probability space mismatch")
    return (probabilities_vector(probabilities) >= threshold).astype(np.int8)


def classification_metrics(labels, probabilities, threshold):
    y = np.asarray(labels)
    pred = apply_threshold(probabilities, threshold).astype(bool)
    if y.shape != pred.shape or not np.isin(y, [0, 1]).all():
        raise ValueError("Expected matching vectors of binary labels and probabilities")
    tp, tn = int(np.sum(pred & (y == 1))), int(np.sum(~pred & (y == 0)))
    fp, fn = int(np.sum(pred & (y == 0))), int(np.sum(~pred & (y == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else None
    return {"threshold": float(threshold), "n": len(y), "positive": tp + fn, "negative": tn + fp,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "confusion_matrix": [[tn, fp], [fn, tp]], "confusion_matrix_order": "[[TN, FP], [FN, TP]]",
            "precision": precision if len(y) else None, "recall": recall,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else (0.0 if len(y) else None),
            "specificity": tn / (tn + fp) if tn + fp else None,
            "sensitivity": recall, "ppv": precision if pred.any() else None}


def select_threshold(labels, probabilities, *, strategy="recall_constraint", target_recall=0.80, threshold=None):
    """Enumerate distinct score boundaries in O(n log n); ties favor higher thresholds.

    This pure optimizer has no split access. The training workflow must attest that
    its inputs come from the independent threshold-validation patient partition.
    """
    p, y = probabilities_vector(probabilities), np.asarray(labels)
    if y.shape != p.shape or not np.isin(y, [0, 1]).all():
        raise ValueError("Expected matching vectors of binary labels and probabilities")
    if strategy not in STRATEGIES:
        raise ValueError("Unknown threshold_strategy")
    if strategy == "fixed":
        chosen = validate_threshold(threshold)
    else:
        if set(np.unique(y)) != {0, 1}:
            raise ValueError("Automatic threshold selection requires both classes in independent validation")
        if strategy == "recall_constraint" and (
                isinstance(target_recall, (bool, np.bool_)) or not isinstance(target_recall, numbers.Real)
                or not np.isfinite(target_recall) or not 0 < target_recall <= 1):
            raise ValueError("target_recall must lie in (0, 1]")
        order = np.argsort(-p, kind="stable")
        scores, sorted_y = p[order], y[order]
        ends = np.r_[np.flatnonzero(np.diff(scores) != 0), len(scores) - 1]
        tp = np.cumsum(sorted_y, dtype=np.int64)[ends]
        count, positives = ends + 1, int(y.sum())
        precision, recall = tp / count, tp / positives
        objective = precision if strategy == "recall_constraint" else 2 * tp / (count + positives)
        eligible = recall >= target_recall if strategy == "recall_constraint" else np.ones(len(ends), bool)
        candidates = np.flatnonzero(eligible)
        # Boundaries descend, so argmax's first occurrence is the highest tied threshold.
        chosen = float(scores[ends[candidates[np.argmax(objective[candidates])]]])
    return {"threshold": chosen, "threshold_strategy": strategy,
            "target_recall": float(target_recall) if strategy == "recall_constraint" else None,
            "selection_metrics": classification_metrics(y, p, chosen),
            "tie_break": "higher_threshold", "classification_rule": "p >= threshold",
            "unit": "window", "zero_predicted_positive_precision": 0.0}


def validate_policy(policy, *, probability_space, horizon_hours):
    if not isinstance(policy, dict):
        raise ValueError("Missing saved threshold policy; retrain into a new output directory")
    validate_threshold(policy.get("threshold"))
    if policy.get("threshold_strategy") not in STRATEGIES:
        raise ValueError("Missing/invalid saved threshold_strategy")
    if policy.get("probability_space") != probability_space:
        raise ValueError("Threshold probability space mismatch")
    if policy.get("horizon_hours") != horizon_hours:
        raise ValueError("Threshold horizon mismatch")
    if policy.get("classification_rule") != "p >= threshold":
        raise ValueError("Unsupported threshold classification rule")
    return policy


def apply_policy(probabilities, policy, *, probability_space, horizon_hours):
    validate_policy(policy, probability_space=probability_space, horizon_hours=horizon_hours)
    return apply_threshold(probabilities, policy["threshold"], probability_space=probability_space,
                           expected_space=policy["probability_space"])
