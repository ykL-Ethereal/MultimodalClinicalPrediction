"""Summarize saved runs without loading models or importing training dependencies."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys


MODELS = ("tcn", "lstm", "gru", "grud", "xgboost", "news2")
METRICS = (
    "status", "n", "positive", "negative", "auprc", "auroc", "brier", "log_loss",
    "threshold", "sensitivity", "specificity", "ppv", "tp", "tn", "fp", "fn",
    "precision", "recall", "f1", "confusion_matrix", "confusion_matrix_order",
)
POLICY_FIELDS = (
    "threshold", "threshold_strategy", "target_recall", "probability_space",
    "horizon_hours", "classification_rule", "unit", "tie_break",
)
SOURCE_FIELDS = (
    "split", "role", "independent_of_model_and_calibrator_fit",
    "selection_metrics_are_optimization_set_statistics", "total_windows", "excluded_windows",
)
PROJECT_ROOT = Path(__file__).resolve().parent


def read_json(path, issues, expected=dict):
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8-sig") as stream:
            value = json.load(stream)
        if not isinstance(value, expected):
            raise ValueError("unexpected JSON structure")
        return value
    except (OSError, ValueError) as exc:
        issues.append(f"{path.name}: {exc}")
        return None


def discover_runs(root):
    """Stop at run directories; never traverse feature caches or artifacts."""
    markers = {"status.json", "run_summary.json", "history.json", "test_report.json"}
    for directory, children, files in os.walk(root):
        if markers.intersection(files):
            children[:] = []
            yield Path(directory)
        else:
            children[:] = sorted(name for name in children if name not in {
                "artifact", "feature_cache", "macbert_cache", "__pycache__",
            })


def history_fields(history, model, summary):
    result = {"best_epoch": summary.get("best_epoch"),
              "best_iteration": summary.get("best_iteration"),
              "best_round": None, "trained_epochs": None, "trained_rounds": None,
              "best_selection_score": None, "best_train_loss": None}
    if model == "xgboost" and isinstance(history, dict):
        scores = history.get("validation", {}).get("auprc_ap", [])
        result["trained_rounds"] = len(scores)
        valid = [(i, score) for i, score in enumerate(scores)
                 if isinstance(score, (int, float)) and math.isfinite(score)]
        if valid:
            best, score = max(valid, key=lambda item: item[1])
            if result["best_iteration"] is None:
                result["best_iteration"] = best
            result["best_selection_score"] = score
    elif isinstance(history, list):
        result["trained_epochs"] = len(history)
        valid = [item for item in history if isinstance(item, dict)
                 and isinstance(item.get("selection_score"), (int, float))
                 and math.isfinite(item["selection_score"])]
        if valid:
            best = max(valid, key=lambda item: item["selection_score"])
            if result["best_epoch"] is None:
                result["best_epoch"] = best.get("epoch")
            result["best_selection_score"] = best["selection_score"]
            result["best_train_loss"] = best.get("train_loss")
    if result["best_iteration"] is not None:
        result["best_round"] = result["best_iteration"] + 1
    return result


def metric_fields(row, prefix, metrics):
    metrics = metrics if isinstance(metrics, dict) else {}
    for key in METRICS:
        row[f"{prefix}_{key}"] = metrics.get(key)


def threshold_policy(directory, summary, reports, issues):
    """Read the saved decision policy, never infer one from training settings."""
    policy = read_json(directory / "threshold.json", issues)
    if policy is None:
        policy = summary.get("threshold_policy")
    if not isinstance(policy, dict):
        policy = next((report["threshold_policy"] for report in reports.values()
                       if isinstance(report.get("threshold_policy"), dict)), None)
    if policy is None:
        policy = read_json(directory / "artifact" / "threshold.json", issues)
    return policy if isinstance(policy, dict) else {}


def policy_fields(row, policy):
    for key in POLICY_FIELDS:
        # The horizon in config describes the run; this one describes the policy.
        column = key if key.startswith("threshold") or key == "target_recall" else f"threshold_{key}"
        row[column] = policy.get(key)
    source = policy.get("source") or {}
    for key in SOURCE_FIELDS:
        row[f"threshold_source_{key}"] = source.get(key)
    metric_fields(row, "threshold_selection", policy.get("selection_metrics"))


def extract_run(directory):
    issues = []
    config = read_json(directory / "config.json", issues) or {}
    status = read_json(directory / "status.json", issues) or {}
    summary = read_json(directory / "run_summary.json", issues) or {}
    history = read_json(directory / "history.json", issues, (dict, list))
    model_config, training = config.get("model", {}), config.get("training", {})
    model = summary.get("vital_encoder") or summary.get("model") or status.get("model")
    model = model or model_config.get("vital_encoder") or directory.parent.name
    # Validation-only jobs use an encoder in config but do not train that encoder.
    if model not in MODELS or directory.parent.name == "validation":
        return None
    vital_only = summary.get("vital_only", model_config.get("vital_only"))
    if vital_only is None:
        vital_only = directory.parent.parent.name == "vital_only"
    row = {
        "model": model, "mode": "vital_only" if vital_only else "multimodal",
        "run_id": directory.name, "status": status.get("status", "unknown"),
        "horizon_hours": config.get("data", {}).get("horizon"),
        "seed": summary.get("seed", training.get("seed")),
        "device": summary.get("device", training.get("device")),
        "precision": summary.get("precision", training.get("precision")),
        "selection_metric": summary.get("selection_metric"),
        **history_fields(history, model, summary),
        "parameters": summary.get("parameters"),
        "model_version": summary.get("model_version"),
    }
    performance = summary.get("performance", {})
    row["training_seconds"] = performance.get("training_seconds")
    row["total_seconds"] = performance.get("prepare_train_evaluate_seconds",
                                            performance.get("evaluate_seconds"))
    reports = {}
    for split in ("validation", "test"):
        report = read_json(directory / f"{split}_report.json", issues)
        report = report if report is not None else summary.get("reports", {}).get(split)
        if not isinstance(report, dict):
            report = {}
        reports[split] = report
        if not report and row["status"] == "completed":
            issues.append(f"missing {split} report")
        metrics = report.get("metrics", {})
        metrics = metrics if isinstance(metrics, dict) else {}
        metric_fields(row, split, metrics)
        metric_fields(row, f"{split}_fixed_0_5", report.get("fixed_0_5"))
        for key in ("calibrated", "invalid_inputs", "prediction_seconds", "metric_unit", "validation_role"):
            row[f"{split}_{key}"] = report.get(key)
        warning = report.get("continuous_warning_metrics")
        for key in ("status", "missing_fields", "reason"):
            row[f"{split}_continuous_warning_{key}"] = (
                warning.get(key) if isinstance(warning, dict) else warning if key == "status" else None)
        n, positive = metrics.get("n"), metrics.get("positive")
        row[f"{split}_positive_rate"] = positive / n if n and positive is not None else None
    policy = threshold_policy(directory, summary, reports, issues)
    policy_fields(row, policy)
    artifact = directory / "artifact"
    row.update(artifact_exists=artifact.is_dir(), artifact_dir=str(artifact.resolve()),
               run_dir=str(directory.resolve()), error_type=status.get("error_type"),
               error=status.get("error"))
    if row["status"] == "completed" and not artifact.is_dir():
        issues.append("missing artifact directory")
    row["extraction_issues"] = "; ".join(issues)
    return {"summary": row, "reports": reports, "threshold_policy": policy}


def collect_results(root, *, models=None, vital_only=False, latest=False):
    records = []
    for directory in sorted(discover_runs(root)):
        record = extract_run(directory)
        if record is None:
            continue
        row = record["summary"]
        if models and row["model"] not in models:
            continue
        if vital_only and row["mode"] != "vital_only":
            continue
        records.append(record)
    if latest:
        selected = {}
        for record in records:
            row = record["summary"]
            key = (row["mode"], row["model"], row["horizon_hours"], row["seed"])
            if key not in selected or row["run_id"] > selected[key]["summary"]["run_id"]:
                selected[key] = record
        records = list(selected.values())
    return sorted(records, key=lambda item: (item["summary"]["mode"],
                  item["summary"]["model"], item["summary"]["run_id"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Extract saved training results into CSV and JSON.")
    parser.add_argument("--runs-dir", type=Path, default=PROJECT_ROOT / "runs",
                        help="Runs root or one run directory (default: project runs/)")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs" / "results",
                        help="Summary output directory (default: project runs/results/)")
    parser.add_argument("--models", nargs="+", choices=MODELS)
    parser.add_argument("--vital-only", action="store_true")
    parser.add_argument("--latest", action="store_true",
                        help="Keep newest run per mode/model/horizon/seed, including unfinished runs")
    args = parser.parse_args(argv)
    if not args.runs_dir.is_dir():
        parser.error(f"Runs directory does not exist: {args.runs_dir}")
    records = collect_results(args.runs_dir, models=args.models,
                              vital_only=args.vital_only, latest=args.latest)
    if not records:
        print("No matching runs found; no output written.", file=sys.stderr)
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path, json_path = (args.output_dir / "training_results.csv",
                           args.output_dir / "training_results.json")
    rows = [record["summary"] for record in records]
    # BOM lets Excel recognize UTF-8 paths and messages on Windows.
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        # Keep structured metrics (e.g. confusion matrices) machine-readable in CSV.
        writer.writerows({key: json.dumps(value, ensure_ascii=False, allow_nan=False)
                          if isinstance(value, (list, dict)) else value
                          for key, value in row.items()} for row in rows)
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump({"runs_root": str(args.runs_dir.resolve()), "runs": records},
                  stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    display = (("validation_auprc", "Val AUPRC"), ("test_auprc", "Test AUPRC"),
               ("test_auroc", "Test AUROC"), ("test_threshold", "Threshold"),
               ("test_precision", "Precision"), ("test_recall", "Recall"), ("test_f1", "F1"))
    print(f"{'Model':<9} {'Mode':<11} {'Status':<10} "
          + " ".join(f"{title:>11}" for _, title in display))
    for row in rows:
        values = [f"{row[key]:.6f}" if row[key] is not None else "-"
                  for key, _ in display]
        print(f"{row['model']:<9} {row['mode']:<11} {row['status']:<10} "
              + " ".join(f"{value:>11}" for value in values) + f"  {row['run_id']}")
        if row["error"] or row["extraction_issues"]:
            print(f"  NOTE: {row['error'] or ''} {row['extraction_issues']}", file=sys.stderr)
    print(f"\nExtracted {len(rows)} runs. Existing summary files are replaced on each extraction.")
    print(f"CSV:  {csv_path.resolve()}\nJSON: {json_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
