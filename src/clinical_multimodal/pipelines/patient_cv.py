"""Train-only model selection, followed by an explicit frozen finalization step."""
from copy import deepcopy
from pathlib import Path
import gc
import time

import numpy as np
import torch

from ..accelerator import resolve_device, resolve_precision
from ..config import Config
from ..data.fixed_adapter import FixedCohortBundle
from ..data.fixed_dataset import FixedPreparedDataset
from ..data.vital_cache import CachedVitalDataset
from ..evaluation.bootstrap import patient_bootstrap
from ..evaluation.calibration import sigmoid
from ..evaluation.metrics import binary_metrics
from ..evaluation.predictions import save_predictions
from ..evaluation.report import save_report
from ..evaluation.thresholds import select_threshold
from ..io import file_sha256, fingerprint, read_json, write_json
from ..models.predictor import make_risk_model
from ..preprocessing.vital_numeric import VitalNumericPreprocessor
from ..run_logging import create_run_dir, run_logging
from ..training.checkpoint import save_artifact, source_fingerprint
from ..training.trainer import Trainer, collect_predictions, make_loader, seed_everything
from .threshold import validation_partition, partition_record, fit_threshold, evaluation_details


def dataset_identity(bundle):
    """Bind experiments to metadata and tensor file versions, without reading holdout outcomes."""
    files = {name: file_sha256(bundle.root / name) for name in
             ("dataset_schema.json", "sample_index.csv.gz", "patient_splits.csv")}
    files["tensor_file_versions"] = fingerprint([
        (spec["file"], (bundle.root / spec["file"]).stat().st_size,
         (bundle.root / spec["file"]).stat().st_mtime_ns) for spec in bundle.schema["shards"]])
    return files


def patient_folds(bundle, horizon, folds, seed):
    rows = bundle.rows("train")
    frame = bundle.metadata.iloc[rows]
    y, mask = bundle.labels(rows, horizon)
    positive = set(frame.loc[mask & (y == 1), "patient_id"])
    patients = sorted(frame.patient_id.unique())
    rng = np.random.default_rng(seed)
    assigned = [[] for _ in range(folds)]
    for state in (False, True):
        group = np.asarray([p for p in patients if (p in positive) == state])
        if state and len(group) < folds:
            raise ValueError("Too few positive training patients for the requested folds")
        rng.shuffle(group)
        for i, subset in enumerate(np.array_split(group, folds)):
            assigned[i].extend(subset.tolist())
    result = []
    for patient_ids in assigned:
        selected = frame.patient_id.isin(patient_ids).to_numpy()
        train, validation = rows[~selected], rows[selected]
        if not len(train) or not len(validation):
            raise ValueError("Empty patient fold")
        if set(bundle.metadata.iloc[train].patient_id) & set(bundle.metadata.iloc[validation].patient_id):
            raise ValueError("Patient leakage inside CV")
        for part in (train, validation):
            labels, valid = bundle.labels(part, horizon)
            if set(labels[valid]) != {0, 1}:
                raise ValueError("Every CV part must contain both window classes")
        result.append((train, validation))
    return result


def fold_record(bundle, folds):
    return [{role: {"sample_ids_sha256": fingerprint(bundle.metadata.iloc[rows].sample_id.tolist()),
                    "rows": rows.tolist(), "patient_ids": sorted(bundle.metadata.iloc[rows].patient_id.unique())}
             for role, rows in (("train", train), ("selection", val))} for train, val in folds]


def configure_device(config):
    config.training.device = str(resolve_device(config.training.device))
    config.training.precision = resolve_precision(torch.device(config.training.device), config.training.precision)
    seed_everything(config.training.seed, config.training.num_threads, deterministic=config.training.deterministic)


def dataset(bundle, rows, preprocessor, config):
    if config.training.use_numeric_cache:
        return CachedVitalDataset(bundle, rows, preprocessor, config.data.horizon,
            config.training.numeric_cache_dir, defer_numeric=config.training.device.startswith("cuda"))
    return FixedPreparedDataset(bundle, rows, preprocessor, None, config.data.horizon,
                                defer_numeric=config.training.device.startswith("cuda"))


def model_for(config, bundle, processor):
    return make_risk_model(config.model, len(bundle.contract.vital_names), 0, 0,
                           vital_channels=processor.output_vital_channels)


def predictions(model, ds, config):
    return collect_predictions(model, make_loader(ds, config.training.batch_size,
        device=config.training.device, config=config.training), config.training.device, precision=config.training.precision)


def cv_pipeline(config):
    config = deepcopy(config)
    config.validate()
    output = create_run_dir(Path(config.training.output_dir) / "cv", config.model.vital_encoder)
    config.training.output_dir = str(output)
    with run_logging(output, config):
        configure_device(config)
        write_json(output / "config.json", config.to_dict())
        bundle = FixedCohortBundle(config.data.root, vital_only=True)
        folds = patient_folds(bundle, config.data.horizon, config.training.cv_folds, config.training.seed)
        record = fold_record(bundle, folds)
        write_json(output / "cv_partitions.json", record)
        reports = []
        for i, (train, val) in enumerate(folds):
            fold_output = output / f"fold_{i + 1}"
            fold_output.mkdir()
            seed_everything(config.training.seed, config.training.num_threads, deterministic=config.training.deterministic)
            processor = VitalNumericPreprocessor.fit(bundle, train, device=config.training.device)
            write_json(fold_output / "preprocessing.json", processor.to_dict())
            if set(processor.fit_provenance["patient_ids"]) & set(bundle.metadata.iloc[val].patient_id):
                raise ValueError("Preprocessing fit leaked into fold selection patients")
            train_data, val_data = dataset(bundle, train, processor, config), dataset(bundle, val, processor, config)
            model = model_for(config, bundle, processor)
            trainer = Trainer(model, config.training, fold_output, log_prefix=f"fold={i + 1}/{len(folds)}")
            result = trainer.fit(train_data, val_data)
            collected = predictions(model, val_data, config)
            valid = collected["label_mask"] & collected["valid"]
            y, p = collected["targets"][valid], sigmoid(collected["logits"][valid])
            operating = {str(r): select_threshold(y, p, strategy="min_fp_recall", target_recall=r)
                         for r in sorted(set([config.training.target_recall, .8, .85, .9]))}
            report = {"fold": i + 1, "best_epoch": result.best_epoch,
                      "fpr": operating[str(config.training.target_recall)]["selection_metrics"]["fpr"],
                      "operating_points": operating, "metrics": binary_metrics(y, p),
                      "excluded_windows": int((~valid).sum())}
            reports.append(report)
            write_json(fold_output / "selection_report.json", report)
            selection_policy = {**operating[str(config.training.target_recall)],
                                "probability_space": "raw", "horizon_hours": config.data.horizon}
            save_predictions(fold_output / "selection_predictions.csv", collected, config, None, selection_policy)
            del trainer, model, train_data, val_data
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        summary = {"protocol": "train_patient_cv_v1", "config": config.to_dict(),
                   "dataset_identity": dataset_identity(bundle), "code_sha256": source_fingerprint(),
                   "partition_sha256": fingerprint(record), "folds": reports,
                   "selection_metric": "mean_fold_min_fpr_at_target_recall",
                   "mean_fpr": float(np.mean([r["fpr"] for r in reports])),
                   "std_fpr": float(np.std([r["fpr"] for r in reports])),
                   "mean_auprc": float(np.mean([r["metrics"]["auprc"] for r in reports])),
                   "final_epochs": max(1, int(np.floor(np.median([r["best_epoch"] for r in reports]) + .5))),
                   "validation_used_for_selection": False, "test_evaluated": False,
                   "cv_scores_are_model_selection_statistics_not_unbiased_test_estimates": True}
        summary["sha256"] = fingerprint(summary)
        write_json(output / "cv_summary.json", summary)
        return output / "cv_summary.json"


def finalize_cv(summary_path, output_root=None):
    summary_path = Path(summary_path)
    summary = read_json(summary_path)
    unsigned = {k: v for k, v in summary.items() if k != "sha256"}
    if fingerprint(unsigned) != summary.get("sha256") or summary.get("protocol") != "train_patient_cv_v1":
        raise ValueError("Invalid or edited CV summary")
    if summary["code_sha256"] != source_fingerprint():
        raise ValueError("Training code changed since CV; rerun CV before finalization")
    config = Config.from_dict(summary["config"])
    bundle = FixedCohortBundle(config.data.root, vital_only=True)
    if dataset_identity(bundle) != summary["dataset_identity"]:
        raise ValueError("Dataset changed since CV")
    # This marker prevents accidental repeated test inspection for the same selected run.
    lock = summary_path.parent / "finalization_started.json"
    if lock.exists():
        raise FileExistsError("This CV run has already started finalization; inspect finalization_started.json")
    output = create_run_dir(Path(output_root or summary_path.parent.parent.parent) / "final", config.model.vital_encoder)
    with lock.open("x", encoding="utf-8") as stream:
        import json
        json.dump({"output": str(output), "summary_sha256": summary["sha256"]}, stream)
    config.training.epochs = summary["final_epochs"]
    config.training.output_dir = str(output)
    with run_logging(output, config):
        configure_device(config)
        write_json(output / "config.json", config.to_dict())
        started = time.perf_counter()
        rows = bundle.rows("train")
        processor = VitalNumericPreprocessor.fit(bundle, rows, device=config.training.device)
        train_data = dataset(bundle, rows, processor, config)
        model = model_for(config, bundle, processor)
        trainer = Trainer(model, config.training, output)
        if trainer.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(trainer.device)
        result = trainer.fit(train_data, None)
        parts = validation_partition(bundle, config)
        write_json(output / "validation_partitions.json", partition_record(bundle, config, parts))
        collected = predictions(model, dataset(bundle, parts[2], processor, config), config)
        policy = fit_threshold(bundle, config, parts[2], collected, None, parts)
        policy["source"]["model_selection"] = {"protocol": "train_patient_cv", "summary_sha256": summary["sha256"]}
        write_json(output / "threshold.json", policy)
        save_predictions(output / "threshold_validation_predictions.csv", collected, config, None, policy)
        # Persist the frozen decision artifact BEFORE accessing test tensors or evaluating test labels.
        record = {"vital_encoder": config.model.vital_encoder, "vital_only": True,
                  "protocol": "train_patient_cv_then_full_validation", "cv_summary_sha256": summary["sha256"],
                  "dataset_identity": summary["dataset_identity"], "seed": config.training.seed,
                  "best_epoch": result.best_epoch, "selection_metric": result.selection_metric,
                  "pos_weight": result.pos_weight, "device": config.training.device, "precision": config.training.precision,
                  "threshold_policy": policy,
                  "performance": {"training_seconds": time.perf_counter() - started}}
        manifest = save_artifact(output / "artifact", model, config, bundle, processor, None,
                                 training_record=record, threshold_policy=policy)
        reports = {}
        for split in ("validation", "test"):
            if split == "test":
                collected = predictions(model, dataset(bundle, bundle.rows(split), processor, config), config)
            eligible = collected["label_mask"] & collected["valid"]
            y, p = collected["targets"][eligible], sigmoid(collected["logits"][eligible])
            ids = np.asarray([m["patient_id"] for m in collected["metadata"]])[eligible]
            metrics = binary_metrics(y, p, policy["threshold"])
            save_predictions(output / f"{split}_predictions.csv", collected, config, None, policy)
            extra = {"calibrated": False, "invalid_inputs": int((~collected["valid"]).sum()),
                     "validation_role": "threshold_optimization" if split == "validation" else None,
                     **evaluation_details(y, p, policy, bundle)}
            if split == "test" and config.training.bootstrap_replicates:
                extra["patient_bootstrap"] = patient_bootstrap(y, p, ids, policy["threshold"],
                    replicates=config.training.bootstrap_replicates, seed=config.training.seed)
            reports[split] = save_report(output / f"{split}_report.json", metrics, split=split, extra=extra)
        write_json(output / "run_summary.json", {**record, "model_version": manifest["model_version"], "reports": reports,
                   "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(trainer.device))
                   if trainer.device.type == "cuda" else None})
        return output / "artifact"
