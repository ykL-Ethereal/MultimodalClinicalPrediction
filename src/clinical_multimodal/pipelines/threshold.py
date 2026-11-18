"""Patient-held-out decision fitting. Test labels never enter this workflow."""
import numpy as np

from ..evaluation.calibration import sigmoid
from ..evaluation.metrics import binary_metrics
from ..evaluation.thresholds import select_threshold
from ..io import file_sha256, fingerprint


def validation_partition(bundle, config):
    """Split validation patients only; never borrow from train/test when undersized."""
    rows = bundle.rows("validation")
    metadata = bundle.metadata
    if metadata[["patient_id", "split"]].isna().any().any() or (
            metadata.groupby("patient_id").split.nunique() > 1).any():
        raise ValueError("Patient leakage/missing identity across dataset splits")
    if config.training.protocol == "patient_cv":
        config.validate()
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), rows
    auto = config.training.threshold_strategy != "fixed"
    patients = np.array(sorted(metadata.iloc[rows].patient_id.unique()))
    roles = 1 + int(config.training.calibration) + int(auto)
    if len(patients) < roles:
        raise ValueError(f"Independent validation requires at least {roles} validation patients")
    rng = np.random.default_rng(config.training.seed)
    y, mask = bundle.labels(rows, config.data.horizon)
    # Stratify by any positive window for THIS horizon, without reading test labels.
    positive = set(metadata.iloc[rows[mask & (y == 1)]].patient_id)

    def take(pool, fraction):
        selected, remainder = [], []
        for has_positive in (False, True):
            group = np.array([p for p in pool if (p in positive) == has_positive])
            rng.shuffle(group)
            count = min(len(group) - 1, max(1, round(len(group) * fraction))) if len(group) > 1 else 0
            selected.extend(group[:count])
            remainder.extend(group[count:])
        if not selected or not remainder:
            # With singleton strata stratification is impossible. Keep patient
            # independence; downstream fitting still requires both window classes.
            group = pool.copy()
            rng.shuffle(group)
            count = min(len(group) - 1, max(1, round(len(group) * fraction)))
            if len(group) < 2:
                raise ValueError("Insufficient validation patients for independent partitions")
            return group[count:], group[:count]
        return np.array(remainder), np.array(selected)

    threshold_patients = np.empty(0, dtype=str)
    calibration_patients = np.empty(0, dtype=str)
    if auto:
        patients, threshold_patients = take(patients, config.training.threshold_fraction)
    if config.training.calibration:
        patients, calibration_patients = take(patients, config.training.calibration_fraction)
    ids = metadata.iloc[rows].patient_id
    return tuple(rows[ids.isin(group).to_numpy()] for group in
                 (patients, calibration_patients, threshold_patients))


def partition_record(bundle, config, partitions):
    record = {"seed": config.training.seed, "horizon_hours": config.data.horizon,
              "method": ("full_validation_after_train_patient_cv" if config.training.protocol == "patient_cv"
                         else "validation_patient_stratification_by_any_positive_window"),
              "threshold_fraction": 1.0 if config.training.protocol == "patient_cv" else config.training.threshold_fraction,
              "calibration_fraction_of_remaining": 0.0 if config.training.protocol == "patient_cv" else config.training.calibration_fraction,
              "dataset_patient_splits_sha256": file_sha256(bundle.root / "patient_splits.csv"),
              "partitions": {}}
    for name, rows in zip(("selection_validation", "calibration", "threshold_validation"), partitions):
        frame = bundle.metadata.iloc[rows]
        record["partitions"][name] = {"patient_ids": sorted(frame.patient_id.unique().tolist()),
                                       "sample_ids": frame.sample_id.tolist()}
    record["partition_sha256"] = fingerprint(record)
    return record


def fit_threshold(bundle, config, rows, collected, calibrator, partitions):
    """Only accepts the attested held-out rows, and only after model/calibrator freeze."""
    expected = partitions[2] if config.training.threshold_strategy != "fixed" else partitions[0]
    independent = validation_partition(bundle, config)
    if any(not np.array_equal(a, b) for a, b in zip(partitions, independent)):
        raise ValueError("Threshold patient partitions differ from configured validation split")
    if not np.array_equal(rows, expected) or not bundle.metadata.iloc[rows].split.eq("validation").all():
        raise ValueError("Threshold selection requires the designated validation partition; test data forbidden")
    if [str(m["sample_id"]) for m in collected["metadata"]] != bundle.metadata.iloc[rows].sample_id.astype(str).tolist():
        raise ValueError("Threshold predictions do not match validation sample identities/order")
    valid = collected["label_mask"] & collected["valid"]
    logits = collected["logits"][valid]
    if not np.isfinite(logits).all():
        raise ValueError("Threshold selection requires finite model logits")
    p = calibrator.predict(logits) if calibrator else sigmoid(logits)
    policy = select_threshold(collected["targets"][valid], p, strategy=config.training.threshold_strategy,
                              target_recall=config.training.target_recall, threshold=config.training.threshold)
    policy.update(probability_space="calibrated" if calibrator else "raw", horizon_hours=config.data.horizon,
                  source={"split": "validation", "role": "threshold_validation" if len(partitions[2]) else "manual_fixed",
                          "independent_of_model_and_calibrator_fit": bool(len(partitions[2])),
                          "selection_metrics_are_optimization_set_statistics": True,
                          "partition": partition_record(bundle, config, partitions),
                          "eligible_sample_ids_sha256": fingerprint(bundle.metadata.iloc[rows[valid]].sample_id.tolist()),
                          "total_windows": len(rows), "excluded_windows": int((~valid).sum())},
                  research_target_notice="Window recall target is configurable, not a validated clinical standard",
                  alarm_rule={"unit": "window", "merge": "none", "cooldown_minutes": 0,
                              "prediction_interval_minutes": bundle.schema.get("sampling_interval_minutes")})
    return policy


def evaluation_details(labels, probability, policy, bundle):
    return {"threshold_policy": policy,
            "fixed_0_5": binary_metrics(labels, probability, 0.5),
            "metric_unit": "window", "recall_notice": "Window recall is not patient/event detection rate",
            "continuous_warning_metrics": warning_contract(bundle)}


def warning_contract(bundle):
    fields = {"patient_id", "episode_id", "prediction_time", "event_time", "monitor_start", "monitor_end"}
    missing = sorted(fields - set(bundle.metadata.columns))
    return {"status": "not_computed",
                "missing_fields": missing,
                "reason": "Authoritative event times and complete monitoring exposure contract required; "
                          "sampled window times/TTE bounds cannot substitute for monitoring exposure",
                "window_fpr_is_not_false_alarms_per_patient_day": True}


def audit_threshold_splits(config):
    """Metadata-only audit; does not train, calibrate or select a numerical threshold."""
    from ..data.fixed_adapter import FixedCohortBundle
    from ..io import write_json
    from ..run_logging import create_run_dir
    from pathlib import Path
    bundle = FixedCohortBundle(config.data.root, vital_only=config.model.vital_only)
    parts = validation_partition(bundle, config)
    record = partition_record(bundle, config, parts)
    if config.training.protocol == "patient_cv":
        from .patient_cv import patient_folds, fold_record
        folds = patient_folds(bundle, config.data.horizon, config.training.cv_folds, config.training.seed)
        record["train_cv_partitions"] = fold_record(bundle, folds)
        record["train_cv_counts"] = []
        for train, val in folds:
            item = {}
            for role, rows in (("train", train), ("selection", val)):
                y, mask = bundle.labels(rows, config.data.horizon)
                item[role] = {"windows": len(rows), "patients": int(bundle.metadata.iloc[rows].patient_id.nunique()),
                              "positive_windows": int((y[mask] == 1).sum()),
                              "positive_patients": int(bundle.metadata.iloc[rows[mask & (y == 1)]].patient_id.nunique())}
            record["train_cv_counts"].append(item)
    record["counts"] = {}
    for name, rows in [("train", bundle.rows("train")), *zip(
            ("selection_validation", "calibration", "threshold_validation"), parts), ("test", bundle.rows("test"))]:
        y, mask = bundle.labels(rows, config.data.horizon)
        record["counts"][name] = {"patients": int(bundle.metadata.iloc[rows].patient_id.nunique()),
            "windows": len(rows), "labelled": int(mask.sum()), "positive": int((y[mask] == 1).sum())}
    record["continuous_warning_contract"] = warning_contract(bundle)
    record["status"] = "metadata_only_no_threshold_selected_no_tensor_validity_checked"
    output = create_run_dir(Path(config.training.output_dir) / "audits", f"h{config.data.horizon}")
    write_json(output / "threshold_split_audit.json", record)
    return output / "threshold_split_audit.json"
