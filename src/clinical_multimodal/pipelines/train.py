from __future__ import annotations

from pathlib import Path
from copy import deepcopy
import numpy as np
import time
import torch

from ..data.fixed_dataset import FixedPreparedDataset as PreparedDataset
from ..evaluation.calibration import PlattCalibrator, sigmoid
from ..evaluation.metrics import binary_metrics
from ..evaluation.report import save_report
from ..io import write_json
from ..models.predictor import MultimodalRiskModel
from ..profiling import memory_snapshot
from ..training.checkpoint import save_artifact
from ..training.trainer import Trainer, collect_predictions, make_loader, seed_everything
from .prepare import prepare
from ..run_logging import create_run_dir, run_logging


def validation_partition(bundle, config):
    rows = bundle.rows("validation")
    if not config.training.calibration:
        return rows, np.empty(0, dtype=int)
    patients = np.array(sorted(bundle.metadata.iloc[rows]["patient_id"].unique()))
    if len(patients) < 2:
        raise ValueError("Independent calibration requires at least two validation patients")
    rng = np.random.default_rng(config.training.seed)
    rng.shuffle(patients)
    count = min(len(patients) - 1, max(1, round(len(patients) * config.training.calibration_fraction)))
    calibration = bundle.metadata.iloc[rows]["patient_id"].isin(patients[:count]).to_numpy()
    return rows[~calibration], rows[calibration]


def train_pipeline(config):
    config = deepcopy(config)
    config.validate()
    output = create_run_dir(config.training.output_dir, config.model.vital_encoder)
    config.training.output_dir = str(output)
    with run_logging(output, config):
        return _train_pipeline(config)


def _train_pipeline(config):
    config.validate()
    output = Path(config.training.output_dir)
    if (output / "artifact").exists() or (output / "best.pt").exists():
        raise FileExistsError("Run directory contains trained output; choose a new output_dir")
    seed_everything(config.training.seed, config.training.num_threads)
    started = time.perf_counter()
    bundle, preprocessor, backend, cache = prepare(config)
    selection_rows, calibration_rows = validation_partition(bundle, config)
    if len(selection_rows) == 0 or len(bundle.rows("train")) == 0:
        raise ValueError("Training and selection validation must both be nonempty")
    datasets = {"train": PreparedDataset(bundle, bundle.rows("train"), preprocessor, cache, config.data.horizon),
                "validation": PreparedDataset(bundle, selection_rows, preprocessor, cache, config.data.horizon)}
    model = MultimodalRiskModel(len(bundle.contract.vital_names), len(bundle.contract.lab_names), backend.dimension, config.model,
                               vital_channels=getattr(preprocessor, "output_vital_channels", None))
    trainer = Trainer(model, config.training, output)
    if trainer.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(trainer.device)
    train_started = time.perf_counter()
    result = trainer.fit(datasets["train"], datasets["validation"])
    training_seconds = time.perf_counter() - train_started
    calibrator = None
    if len(calibration_rows):
        calibration_data = PreparedDataset(bundle, calibration_rows, preprocessor, cache, config.data.horizon)
        collected = collect_predictions(model, make_loader(calibration_data, config.training.batch_size), trainer.device)
        valid = collected["label_mask"] & collected["valid"]
        calibrator = PlattCalibrator.fit(collected["logits"][valid], collected["targets"][valid])
    reports = {}
    for split in ("validation", "test"):
        rows = selection_rows if split == "validation" else bundle.rows("test")
        if len(rows):
            dataset = datasets["validation"] if split == "validation" else PreparedDataset(bundle, rows, preprocessor, cache, config.data.horizon)
            collected = collect_predictions(model, make_loader(dataset, config.training.batch_size), trainer.device)
            valid = collected["label_mask"] & collected["valid"]
            logits = collected["logits"][valid]
            probabilities = calibrator.predict(logits) if calibrator else sigmoid(logits)
            metrics = binary_metrics(collected["targets"][valid], probabilities, config.training.threshold)
            invalid_count = int((~collected["valid"]).sum())
        else:
            metrics, invalid_count = binary_metrics([], [], config.training.threshold), 0
        reports[split] = save_report(output / f"{split}_report.json", metrics,
             split=split,
             extra={"calibrated": calibrator is not None, "invalid_inputs": invalid_count,
                    "continuous_warning_metrics": "not_computed_requires_event_and_exposure_contract"})
    record = {"vital_encoder": config.model.vital_encoder,
              "parameters": sum(p.numel() for p in model.parameters()),
              "best_epoch": result.best_epoch, "selection_metric": result.selection_metric,
              "pos_weight": result.pos_weight, "seed": config.training.seed,
              "split_rows": {"train": bundle.rows("train").tolist(), "selection_validation": selection_rows.tolist(),
                             "calibration": calibration_rows.tolist(), "test": bundle.rows("test").tolist()},
              "cache": cache.stats(),
              "performance": {"prepare_train_evaluate_seconds": time.perf_counter()-started,
                              "training_seconds": training_seconds, **memory_snapshot(),
                              "training_evaluation_cuda_peak_allocated_bytes":
                              int(torch.cuda.max_memory_allocated(trainer.device)) if trainer.device.type == "cuda" else None}}
    manifest = save_artifact(output / "artifact", model, config, bundle, preprocessor, backend,
                             calibrator=calibrator, training_record=record)
    write_json(output / "run_summary.json", {**record, "model_version": manifest["model_version"], "reports": reports})
    return output / "artifact"

