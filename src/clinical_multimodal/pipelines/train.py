from __future__ import annotations

from pathlib import Path
from copy import deepcopy
import numpy as np
import time
import torch

from ..accelerator import resolve_device, resolve_precision
from ..data.fixed_dataset import FixedPreparedDataset as PreparedDataset
from ..evaluation.calibration import PlattCalibrator, sigmoid
from ..evaluation.metrics import binary_metrics
from ..evaluation.predictions import save_predictions
from ..evaluation.report import save_report
from ..io import write_json
from ..models.predictor import make_risk_model
from ..profiling import memory_snapshot
from ..training.checkpoint import save_artifact
from ..training.trainer import Trainer, collect_predictions, make_loader, seed_everything
from .prepare import prepare
from ..run_logging import create_run_dir, run_logging
from .threshold import validation_partition, partition_record, fit_threshold, evaluation_details


def train_pipeline(config):
    if config.model.vital_encoder == "xgboost":
        from .xgboost import xgboost_pipeline
        return xgboost_pipeline(config)
    if config.model.vital_encoder == "news2":
        from .news2 import news2_pipeline
        return news2_pipeline(config)
    config = deepcopy(config)
    config.validate()
    root = Path(config.training.output_dir) / "vital_only" if config.model.vital_only else config.training.output_dir
    output = create_run_dir(root, config.model.vital_encoder)
    config.training.output_dir = str(output)
    with run_logging(output, config):
        return _train_pipeline(config)


def _train_pipeline(config):
    config.validate()
    output = Path(config.training.output_dir)
    if (output / "artifact").exists() or (output / "best.pt").exists():
        raise FileExistsError("Run directory contains trained output; choose a new output_dir")
    config.training.device = str(resolve_device(config.training.device))
    config.training.precision = resolve_precision(torch.device(config.training.device), config.training.precision)
    if not config.model.vital_only:
        config.text.device = str(resolve_device(config.text.device))
        config.text.precision = resolve_precision(torch.device(config.text.device), config.text.precision)
    write_json(output / "config.json", config.to_dict())
    print(f"training_device={config.training.device} precision={config.training.precision} "
          f"text_device={config.text.device if not config.model.vital_only else 'disabled'} "
          f"deterministic={config.training.deterministic}", flush=True)
    seed_everything(config.training.seed, config.training.num_threads, deterministic=config.training.deterministic)
    started = time.perf_counter()
    bundle, preprocessor, backend, cache = prepare(config)
    partitions = validation_partition(bundle, config)
    selection_rows, calibration_rows, threshold_rows = partitions
    write_json(output / "validation_partitions.json", partition_record(bundle, config, partitions))
    if len(selection_rows) == 0 or len(bundle.rows("train")) == 0:
        raise ValueError("Training and selection validation must both be nonempty")
    dataset_options = {"defer_numeric": torch.device(config.training.device).type == "cuda"}
    datasets = {"train": PreparedDataset(bundle, bundle.rows("train"), preprocessor, cache, config.data.horizon, **dataset_options),
                "validation": PreparedDataset(bundle, selection_rows, preprocessor, cache, config.data.horizon, **dataset_options)}
    model = make_risk_model(config.model, len(bundle.contract.vital_names), len(bundle.contract.lab_names),
                            backend.dimension if backend else 0, vital_channels=preprocessor.output_vital_channels)
    trainer = Trainer(model, config.training, output)
    if trainer.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(trainer.device)
    train_started = time.perf_counter()
    result = trainer.fit(datasets["train"], datasets["validation"])
    training_seconds = time.perf_counter() - train_started
    calibrator = None
    if len(calibration_rows):
        calibration_data = PreparedDataset(bundle, calibration_rows, preprocessor, cache, config.data.horizon, **dataset_options)
        collected = collect_predictions(model, make_loader(calibration_data, config.training.batch_size, device=trainer.device),
                                        trainer.device, precision=trainer.precision)
        valid = collected["label_mask"] & collected["valid"]
        calibrator = PlattCalibrator.fit(collected["logits"][valid], collected["targets"][valid])
    decision_rows = threshold_rows if len(threshold_rows) else selection_rows
    decision_data = PreparedDataset(bundle, decision_rows, preprocessor, cache, config.data.horizon, **dataset_options)
    decision_predictions = collect_predictions(model, make_loader(decision_data, config.training.batch_size,
        device=trainer.device), trainer.device, precision=trainer.precision)
    policy = fit_threshold(bundle, config, decision_rows, decision_predictions, calibrator, partitions)
    write_json(output / "threshold.json", policy)
    save_predictions(output / "threshold_validation_predictions.csv", decision_predictions, config, calibrator, policy)
    reports = {}
    for split in ("validation", "test"):
        prediction_started = time.perf_counter()
        rows = selection_rows if split == "validation" else bundle.rows("test")
        if len(rows):
            dataset = datasets["validation"] if split == "validation" else PreparedDataset(bundle, rows, preprocessor, cache, config.data.horizon, **dataset_options)
            collected = collect_predictions(model, make_loader(dataset, config.training.batch_size, device=trainer.device),
                                            trainer.device, precision=trainer.precision)
            valid = collected["label_mask"] & collected["valid"]
            logits = collected["logits"][valid]
            probabilities = calibrator.predict(logits) if calibrator else sigmoid(logits)
            metrics = binary_metrics(collected["targets"][valid], probabilities, policy["threshold"])
            invalid_count = int((~collected["valid"]).sum())
        else:
            metrics, invalid_count = binary_metrics([], [], policy["threshold"]), 0
            valid, probabilities = np.empty(0, bool), np.empty(0)
            collected = {"logits": np.empty(0), "targets": np.empty(0), "metadata": [],
                         "label_mask": np.empty(0, bool), "valid": np.empty(0, bool)}
        save_predictions(output / f"{split}_predictions.csv", collected, config, calibrator, policy)
        reports[split] = save_report(output / f"{split}_report.json", metrics,
             split=split,
             extra={"calibrated": calibrator is not None, "invalid_inputs": invalid_count,
                    "prediction_seconds": time.perf_counter() - prediction_started,
                    "validation_role": "early_stopping_selection" if split == "validation" else None,
                    **evaluation_details(collected["targets"][valid], probabilities, policy, bundle)})
    record = {"vital_encoder": config.model.vital_encoder,
              "vital_only": config.model.vital_only,
              "device": str(trainer.device), "precision": trainer.precision,
              "gpu_name": torch.cuda.get_device_name(trainer.device) if trainer.device.type == "cuda" else None,
              "text_device": str(backend.device) if backend else None, "text_precision": backend.precision if backend else None,
              "deterministic": config.training.deterministic,
              "parameters": sum(p.numel() for p in model.parameters()),
              "best_epoch": result.best_epoch, "selection_metric": result.selection_metric,
              "pos_weight": result.pos_weight, "seed": config.training.seed,
              "split_rows": {"train": bundle.rows("train").tolist(), "selection_validation": selection_rows.tolist(),
                             "calibration": calibration_rows.tolist(), "threshold_validation": threshold_rows.tolist(),
                             "test": bundle.rows("test").tolist()},
              "threshold_policy": policy,
              "cache": cache.stats() if cache else None,
              "performance": {"prepare_train_evaluate_seconds": time.perf_counter()-started,
                              "training_seconds": training_seconds, **memory_snapshot(),
                              "training_evaluation_cuda_peak_allocated_bytes":
                              int(torch.cuda.max_memory_allocated(trainer.device)) if trainer.device.type == "cuda" else None}}
    manifest = save_artifact(output / "artifact", model, config, bundle, preprocessor, backend,
                             calibrator=calibrator, training_record=record, threshold_policy=policy)
    write_json(output / "run_summary.json", {**record, "model_version": manifest["model_version"], "reports": reports})
    return output / "artifact"

