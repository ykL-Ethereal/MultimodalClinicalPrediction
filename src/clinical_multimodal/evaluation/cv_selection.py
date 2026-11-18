"""Rank signed Train-only CV summaries without reading validation/test reports."""
from pathlib import Path
import math

from ..io import fingerprint, read_json


RANKING_RULE = "lower mean FPR, lower fold SD, higher mean AP, model name, summary path"


def comparison_contract(summary):
    t = summary["config"]["training"]
    return {"dataset": summary["dataset_identity"], "horizon": summary["config"]["data"]["horizon"],
            "partitions": summary["partition_sha256"], "target_recall": t["target_recall"],
            "batch_size": t["batch_size"], "steps_per_epoch": t["steps_per_epoch"],
            "epochs": t["epochs"], "patience": t["patience"], "code_sha256": summary["code_sha256"],
            "budget_type": "full_cohort" if t["sampling"] == "original" else "fixed_steps"}


def compare_summaries(paths):
    groups = {}
    for path in paths:
        path = Path(path).resolve()
        s = read_json(path)
        if fingerprint({k: v for k, v in s.items() if k != "sha256"}) != s.get("sha256"):
            raise ValueError(f"Edited CV summary: {path}")
        if (s.get("protocol") != "train_patient_cv_v1" or s.get("test_evaluated") is not False
                or s.get("validation_used_for_selection") is not False):
            raise ValueError("Selection requires Train-only CV summaries")
        for key in ("mean_fpr", "std_fpr", "mean_auprc"):
            value = s[key]
            if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid CV selection metric: {key}")
        identity = comparison_contract(s)
        group = groups.setdefault(fingerprint(identity), {"comparison_contract": identity, "candidates": []})
        group["candidates"].append({"summary": str(path), "summary_sha256": s["sha256"],
            "model": s["config"]["model"]["vital_encoder"], "sampling": s["config"]["training"]["sampling"],
            "mean_fpr": s["mean_fpr"], "std_fpr": s["std_fpr"], "mean_auprc": s["mean_auprc"],
            "final_epochs": s["final_epochs"]})
    for group in groups.values():
        group["candidates"].sort(key=lambda r: (r["mean_fpr"], r["std_fpr"], -r["mean_auprc"], r["model"], r["summary"]))
        group["ranking_rule"] = RANKING_RULE
    return groups
