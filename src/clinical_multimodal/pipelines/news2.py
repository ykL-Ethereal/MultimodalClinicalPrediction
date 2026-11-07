"""Evaluate a fixed NEWS2 baseline on the existing patient splits without fitting."""
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from ..data.fixed_adapter import FixedCohortBundle
from ..evaluation.report import save_report
from ..inference.news2 import NEWS2Predictor
from ..io import write_json
from ..models.news2 import load_rules
from ..run_logging import create_run_dir, run_logging


def score_metrics(labels, scores, threshold):
    """Rank/discrimination and alarm metrics on raw scores, never probability losses."""
    y, s = np.asarray(labels), np.asarray(scores, dtype=float)
    if y.ndim != 1 or y.shape != s.shape or not np.isin(y, [0, 1]).all() or not np.isfinite(s).all():
        raise ValueError("NEWS2 metrics require finite scores and matching binary labels")
    if threshold is not None and (type(threshold) not in (int, float) or not np.isfinite(threshold)):
        raise ValueError("Invalid NEWS2 alarm threshold")
    n = len(y)
    positive, negative = int((y == 1).sum()), int((y == 0).sum())
    pred = s >= threshold if threshold is not None else np.zeros(n, bool)
    tp, tn = int(np.sum(pred & (y == 1))), int(np.sum(~pred & (y == 0)))
    both = positive > 0 and negative > 0
    return {"n": n, "positive": positive, "negative": negative,
        "auroc": float(roc_auc_score(y, s)) if both else None,
        "auprc": float(average_precision_score(y, s)) if both else None,
        "threshold": threshold, "tp": tp if threshold is not None else None,
        "tn": tn if threshold is not None else None,
        "fp": int(np.sum(pred & (y == 0))) if threshold is not None else None,
        "fn": int(np.sum(~pred & (y == 1))) if threshold is not None else None,
        "sensitivity": tp / positive if positive and threshold is not None else None,
        "specificity": tn / negative if negative and threshold is not None else None,
        "ppv": tp / int(pred.sum()) if pred.any() and threshold is not None else None,
        "brier": None, "log_loss": None, "calibration_curve": [],
        "status": "no_valid_labels" if not n else ("ok" if both else "single_class")}


def evaluate_news2(predictor, bundle, output, *, split="test", batch_size=256, predictions_path=None):
    rows = bundle.rows(split)
    y, label_mask = bundle.labels(rows, predictor.horizon)
    scores, valid = np.zeros(len(rows)), np.zeros(len(rows), bool)
    normalized_scores = np.zeros(len(rows))
    partial_inputs = 0
    missing_counts = {name: 0 for name in predictor.model.rules["parameters"]}
    stream = None
    if predictions_path is not None:
        path = Path(predictions_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        stream = path.open("w", encoding="utf-8")
    offset = 0
    try:
        for batch in predictor.iter_predict(bundle, rows=rows, batch_size=batch_size):
            for prediction in batch:
                if stream:
                    stream.write(json.dumps(prediction, ensure_ascii=False, allow_nan=False) + "\n")
                if prediction["score"] is not None:
                    scores[offset], valid[offset] = prediction["score"], True
                    normalized_scores[offset] = prediction["normalized_score"]
                    partial_inputs += bool(prediction["missing_parameters"])
                for name in prediction["missing_parameters"]:
                    missing_counts[name] += 1
                offset += 1
    finally:
        if stream:
            stream.close()
    included = valid & label_mask
    return save_report(output, score_metrics(y[included], scores[included], predictor.model.rules["alarm_threshold"]),
        split=split, extra={"model": "news2", "variant": "reduced", "score_type": "raw_rule_score",
            "normalized_score_type": "available_max_normalized_rule_score",
            "normalized_metrics": score_metrics(y[included], normalized_scores[included], None),
            "calibrated": False, "total_samples": len(rows), "invalid_inputs": int((~valid).sum()),
            "partial_inputs": partial_inputs,
            "missing_parameter_counts": missing_counts,
            "omitted_parameters": predictor.model.rules["omitted_parameters"],
            "continuous_warning_metrics": "not_computed_requires_event_and_exposure_contract"})


def news2_pipeline(config):
    config = deepcopy(config)
    config.model.vital_encoder = "news2"
    config.validate()
    output = create_run_dir(config.training.output_dir, "news2")
    config.training.output_dir = str(output)
    with run_logging(output, config):
        started = time.perf_counter()
        bundle = FixedCohortBundle(config.data.root)
        if config.data.horizon not in bundle.horizons:
            raise ValueError("Unsupported NEWS2 evaluation horizon")
        rules = load_rules(config.news2.rules_file)
        predictor = NEWS2Predictor.save(output / "artifact", rules, bundle.contract, config.data.horizon)
        print(f"NEWS2 reduced baseline: parameters={list(rules['parameters'])}; "
              f"alarm >= {rules['alarm_threshold']}; no fitting", flush=True)
        reports = {}
        for split in ("validation", "test"):
            reports[split] = evaluate_news2(predictor, bundle, output / f"{split}_report.json", split=split,
                batch_size=config.news2.batch_size, predictions_path=output / f"{split}_predictions.jsonl")
            print(f"NEWS2 {split}: {reports[split]['metrics']}", flush=True)
        write_json(output / "run_summary.json", {"model": "news2", "variant": "reduced", "parameters": 0,
            "fitted": False, "model_version": predictor.model.version, "reports": reports,
            "unused_training_options": ["epochs", "calibration", "threshold", "device"],
            "performance": {"evaluate_seconds": time.perf_counter() - started}})
    return output / "artifact"
