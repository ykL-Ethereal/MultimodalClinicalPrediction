"""Patient-split XGBoost baseline with disk batches and CPU external-memory hist."""
from copy import deepcopy
from pathlib import Path
import time
import numpy as np

from ..evaluation.calibration import PlattCalibrator, sigmoid
from ..evaluation.metrics import binary_metrics
from ..evaluation.predictions import save_predictions
from ..evaluation.report import save_report
from ..io import write_json
from ..models.vital_xgboost import flatten_vitals, predict_margins, require_xgboost
from ..profiling import memory_snapshot
from ..run_logging import create_run_dir, run_logging
from ..training.checkpoint import save_artifact
from ..training.losses import training_pos_weight
from .prepare import prepare
from .threshold import validation_partition, partition_record, fit_threshold, evaluation_details


def cache_features(bundle, preprocessor, rows, horizon, directory, batch_size):
    """Write only eligible rows, in fixed order, without a full-cohort dense array."""
    directory.mkdir(parents=True, exist_ok=False)
    paths, labels, sample_rows = [], [], []
    for start in range(0, len(rows), batch_size):
        selected = rows[start:start + batch_size]
        inputs = preprocessor.transform(bundle, selected)
        y, mask = bundle.labels(selected, horizon)
        valid = mask & inputs["available"][:, 0]
        if not valid.any():
            continue
        x_path, y_path = directory / f"x_{len(paths):06d}.npy", directory / f"y_{len(paths):06d}.npy"
        np.save(x_path, flatten_vitals(inputs["vitals"][valid]), allow_pickle=False)
        np.save(y_path, y[valid].astype(np.float32), allow_pickle=False)
        paths.append((x_path, y_path))
        labels.append(y[valid])
        sample_rows.append(selected[valid])
    return paths, np.concatenate(labels) if labels else np.empty(0), np.concatenate(sample_rows) if sample_rows else np.empty(0, int)


def external_matrix(paths, cache_prefix, config, ref=None):
    xgb = require_xgboost()

    class Batches(xgb.DataIter):
        def __init__(self):
            self.index = 0
            super().__init__(cache_prefix=str(cache_prefix), release_data=True)

        def reset(self):
            self.index = 0

        def next(self, input_data):
            if self.index == len(paths):
                return False
            x_path, y_path = paths[self.index]
            input_data(data=np.load(x_path, mmap_mode="r", allow_pickle=False),
                       label=np.load(y_path, allow_pickle=False))
            self.index += 1
            return True

    return xgb.ExtMemQuantileDMatrix(Batches(), max_bin=config.max_bin, ref=ref, nthread=config.nthread)


def collect_xgboost(model, bundle, preprocessor, rows, config):
    logits, valid = [], []
    for start in range(0, len(rows), config.xgboost.batch_size):
        selected = rows[start:start + config.xgboost.batch_size]
        inputs = preprocessor.transform(bundle, selected)
        logits.append(predict_margins(model, inputs["vitals"]))
        valid.append(inputs["available"][:, 0])
    y, mask = bundle.labels(rows, config.data.horizon)
    return {"logits": np.concatenate(logits) if logits else np.empty(0), "targets": y, "label_mask": mask,
            "valid": np.concatenate(valid) if valid else np.empty(0, bool),
            "metadata": bundle.metadata.iloc[rows][["sample_id"]].to_dict("records")}


def xgboost_pipeline(config):
    config = deepcopy(config)
    config.validate()
    if not config.model.vital_only or config.model.vital_encoder != "xgboost":
        raise ValueError("XGBoost requires vital_only mode and the xgboost model")
    config.training.device, config.training.precision = "cpu", "float32"
    output = create_run_dir(Path(config.training.output_dir) / "vital_only", "xgboost")
    config.training.output_dir = str(output)
    with run_logging(output, config):
        return _train(config, output)


def _train(config, output):
    from sklearn.metrics import average_precision_score
    xgb = require_xgboost()
    started = time.perf_counter()
    bundle, preprocessor, _, _ = prepare(config)
    partitions = validation_partition(bundle, config)
    selection, calibration, threshold_rows = partitions
    write_json(output / "validation_partitions.json", partition_record(bundle, config, partitions))
    cache_root = output / "feature_cache"
    caches = {}
    for split, rows in (("train", bundle.rows("train")), ("validation", selection)):
        paths, labels, selected = cache_features(bundle, preprocessor, rows, config.data.horizon,
                                                cache_root / split, config.xgboost.batch_size)
        if set(np.unique(labels)) != {0, 1}:
            raise ValueError(f"XGBoost requires both classes among valid {split} samples")
        caches[split] = (paths, labels, selected)
        print(f"{split}: cached {len(selected)}/{len(rows)} valid rows in {len(paths)} batches", flush=True)
    cfg = config.xgboost
    weight = training_pos_weight(caches["train"][1], np.ones(len(caches["train"][1]), bool)) if config.training.class_weight else None
    train_started = time.perf_counter()
    train = external_matrix(caches["train"][0], cache_root / "train_pages", cfg)
    validation = external_matrix(caches["validation"][0], cache_root / "validation_pages", cfg, ref=train)
    params = {"objective": "binary:logistic", "tree_method": "hist", "device": "cpu",
              "max_depth": cfg.max_depth, "eta": cfg.learning_rate, "min_child_weight": cfg.min_child_weight,
              "subsample": cfg.subsample, "colsample_bytree": cfg.colsample_bytree, "lambda": cfg.reg_lambda,
              "max_bin": cfg.max_bin, "nthread": cfg.nthread, "seed": config.training.seed,
              "scale_pos_weight": weight if weight is not None else 1., "disable_default_eval_metric": 1}

    def average_precision(probability, matrix):
        return "auprc_ap", float(average_precision_score(matrix.get_label(), probability))

    history = {}
    model = xgb.train(params, train, num_boost_round=cfg.num_boost_round,
                      evals=[(validation, "validation")], custom_metric=average_precision, maximize=True,
                      early_stopping_rounds=cfg.early_stopping_rounds, evals_result=history, verbose_eval=True)
    model.set_attr(prediction_nthread=str(cfg.nthread))
    training_seconds = time.perf_counter() - train_started
    # Release page files before reporting memory / leaving the run.
    del validation, train
    write_json(output / "history.json", history)
    write_json(output / "xgboost_params.json", params)
    calibrator = None
    if len(calibration):
        collected = collect_xgboost(model, bundle, preprocessor, calibration, config)
        valid = collected["label_mask"] & collected["valid"]
        calibrator = PlattCalibrator.fit(collected["logits"][valid], collected["targets"][valid])
    decision_rows = threshold_rows if len(threshold_rows) else selection
    collected = collect_xgboost(model, bundle, preprocessor, decision_rows, config)
    policy = fit_threshold(bundle, config, decision_rows, collected, calibrator, partitions)
    write_json(output / "threshold.json", policy)
    save_predictions(output / "threshold_validation_predictions.csv", collected, config, calibrator, policy)
    reports = {}
    for split, rows in (("validation", selection), ("test", bundle.rows("test"))):
        prediction_started = time.perf_counter()
        collected = collect_xgboost(model, bundle, preprocessor, rows, config)
        valid = collected["label_mask"] & collected["valid"]
        logits = collected["logits"][valid]
        probability = calibrator.predict(logits) if calibrator else sigmoid(logits)
        metrics = binary_metrics(collected["targets"][valid], probability, policy["threshold"])
        save_predictions(output / f"{split}_predictions.csv", collected, config, calibrator, policy)
        reports[split] = save_report(output / f"{split}_report.json", metrics, split=split,
                                    extra={"calibrated": calibrator is not None,
                                           "invalid_inputs": int((~collected["valid"]).sum()),
                                           "prediction_seconds": time.perf_counter() - prediction_started,
                                           "validation_role": "early_stopping_selection" if split == "validation" else None,
                                           **evaluation_details(collected["targets"][valid], probability, policy, bundle)})
    record = {"vital_encoder": "xgboost", "vital_only": True, "device": "cpu", "precision": "float32",
              "feature_shape": [120, 20], "feature_count": 2400, "feature_mode": "flatten",
              "best_iteration": model.best_iteration, "selection_metric": "auprc_ap",
              "seed": config.training.seed, "pos_weight": weight,
              "matrix": "ExtMemQuantileDMatrix", "xgboost_version": xgb.__version__,
              "split_rows": {"train": bundle.rows("train").tolist(), "selection_validation": selection.tolist(),
                             "calibration": calibration.tolist(), "threshold_validation": threshold_rows.tolist(),
                             "test": bundle.rows("test").tolist()},
              "threshold_policy": policy,
              "eligible_training_rows": caches["train"][2].tolist(),
              "feature_cache_bytes": sum(p.stat().st_size for p in cache_root.rglob("*.npy")),
              "performance": {"training_seconds": training_seconds,
                              "prepare_train_evaluate_seconds": time.perf_counter() - started, **memory_snapshot()}}
    manifest = save_artifact(output / "artifact", model, config, bundle, preprocessor, None,
                             calibrator=calibrator, training_record=record, threshold_policy=policy)
    write_json(output / "run_summary.json", {**record, "model_version": manifest["model_version"], "reports": reports})
    return output / "artifact"
