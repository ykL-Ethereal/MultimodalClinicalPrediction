"""Vital-only isolation, 20-channel parity, artifacts and flattened XGBoost inputs."""
from pathlib import Path
from unittest.mock import patch
import importlib.util
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from clinical_multimodal.config import Config
from clinical_multimodal.data.fixed_adapter import FixedCohortBundle
from clinical_multimodal.inference.predictor import Predictor
from clinical_multimodal.io import read_json
from clinical_multimodal.models.vital_xgboost import flatten_vitals
from clinical_multimodal.pipelines.evaluate import evaluate_pipeline
from clinical_multimodal.pipelines.train import train_pipeline
from clinical_multimodal.preprocessing.fixed_numeric import FixedNumericPreprocessor
from clinical_multimodal.preprocessing.vital_numeric import VitalNumericPreprocessor
from clinical_multimodal.training.checkpoint import load_artifact, vital_input_layout
from tests.fixed_fixture import make_fixed
from tests.helpers import TestWorkspace


@pytest.fixture
def workspace():
    with TestWorkspace() as directory:
        yield Path(directory)


def configuration(workspace, model="tcn"):
    cfg = Config()
    cfg.data.root = str(workspace / "data")
    cfg.model.vital_only = True
    cfg.model.vital_encoder = model
    cfg.model.embedding_dim = 8
    cfg.model.tcn_channels = 4
    cfg.model.rnn_hidden = 4
    cfg.model.dropout = 0
    cfg.training.models = [model]
    cfg.training.epochs = cfg.training.patience = cfg.training.num_threads = 1
    cfg.training.batch_size = 2
    cfg.training.device = "cpu"
    cfg.training.threshold_strategy = "fixed"  # One validation patient in the tiny fixture.
    cfg.training.threshold = 0.5
    cfg.training.output_dir = str(workspace / "runs")
    cfg.text.model_name = str(workspace / "nonexistent-text-model")
    cfg.text.device = "cuda:999"  # Ignored by vital-only runs.
    cfg.xgboost.num_boost_round = 3
    cfg.xgboost.early_stopping_rounds = 1
    cfg.xgboost.batch_size = 1
    cfg.xgboost.max_bin = 8
    cfg.xgboost.nthread = 1
    cfg.xgboost.min_child_weight = 0
    return cfg.validate()


def remove_excluded_inputs(root):
    # root is a generated fixture, never the production data package.
    for name in ("lab_feature_catalog.csv", "note_text.jsonl", "note_index.npz"):
        (root / name).unlink()
    for path in (root / "tensors").glob("*.npz"):
        with np.load(path, allow_pickle=False) as z:
            a = {key: z[key] for key in ("vital_features", "vital_observed_mask", "prediction_second", "y")}
        np.savez_compressed(path, **a)
    schema = read_json(root / "dataset_schema.json")
    for key in ("lab_feature_count", "lab_channels", "static_feature_order", "lab_lookback_hours", "notes"):
        schema.pop(key)
    (root / "dataset_schema.json").write_text(json.dumps(schema), encoding="utf-8")


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_vital_preprocessing_matches_multimodal(workspace, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    root = make_fixed(workspace / "data")
    full = FixedCohortBundle(root)
    vital = FixedCohortBundle(root, vital_only=True)
    # Include forward-filled high-frequency values and available but unobserved BP.
    for bundle in (full, vital):
        a = bundle.shard(0)
        a["vital_features"][:, 4:8, 4:8] = 1
        a["vital_observed_mask"][:, 4:8, :4] = False
        a["vital_observed_mask"][:, 4:8, 4:] = False
    expected = FixedNumericPreprocessor.fit(full, full.rows("train"), device=device)
    processor = VitalNumericPreprocessor.fit(vital, vital.rows("train"), device=device)
    np.testing.assert_allclose(expected.vital_mean, processor.vital_mean)
    np.testing.assert_allclose(expected.vital_std, processor.vital_std)
    actual = processor.transform(vital, [0, 1])
    np.testing.assert_array_equal(actual["vitals"], expected.transform(full, [0, 1])["vitals"])
    raw = vital.get_numeric([0, 1])
    tensors = processor.transform_tensors({k: torch.as_tensor(raw[k], device=device)
                                          for k in ("vital_features", "vital_observed_mask")})
    for key in actual:
        np.testing.assert_allclose(actual[key], tensors[key].cpu().numpy(), atol=1e-6)
    with pytest.raises(ValueError, match="training rows"):
        VitalNumericPreprocessor.fit(vital, vital.rows("validation"))


@pytest.mark.parametrize("encoder", ["tcn", "lstm", "gru", "grud"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_train_predict_without_excluded_files(workspace, encoder, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    root = make_fixed(workspace / "data")
    remove_excluded_inputs(root)
    cfg = configuration(workspace, encoder)
    cfg.training.device = device
    cfg.training.precision = "auto"
    with patch("clinical_multimodal.pipelines.prepare.make_backend", side_effect=AssertionError("text initialized")), \
         patch("clinical_multimodal.inference.predictor.make_backend", side_effect=AssertionError("text initialized")), \
         patch.object(FixedCohortBundle, "materialize", side_effect=AssertionError("notes read")):
        artifact = train_pipeline(cfg)
        predictor = Predictor(artifact, device=device)
        result = predictor.predict(root, rows=[5, 0], diagnostics=True)
        reloaded = Predictor(artifact, device="cpu").predict(root, rows=[5, 0], batch_size=1)
    np.testing.assert_allclose([r["risk"] for r in result], [r["risk"] for r in reloaded], atol=.01, rtol=.01)
    assert [r["sample_id"] for r in result] == ["5", "0"]
    assert all(r["used_modalities"] == ["vitals"] for r in result)
    assert not (artifact / "text_backbone").exists()
    assert set(predictor.model._modules) == {"vital", "head"}
    assert artifact.parent.parent.parent.name == "vital_only"
    report = evaluate_pipeline(artifact, root, workspace / "evaluation.json", device="cpu")
    assert report["metrics"]["n"] == 2
    predictions = pd.read_csv(artifact.parent / "test_predictions.csv")
    saved = predictor.predict(root, rows=[4, 5])
    np.testing.assert_allclose(predictions.risk, [r["risk"] for r in saved], atol=.01, rtol=.01)
    assert len(vital_input_layout()["feature_names"]) == 2400
    bundle = FixedCohortBundle(root, vital_only=True)
    a = bundle.shard(0)
    a["vital_features"][:] = 0
    a["vital_features"][:, :, 4:8] = 1
    a["vital_observed_mask"][:] = False
    invalid = predictor.predict(bundle, rows=[0])[0]
    assert invalid["risk"] is None and invalid["status"] == "invalid_input_no_vitals"
    assert invalid["used_modalities"] == []


def test_flatten_preserves_all_positions_and_mask_zeros():
    values = np.arange(2 * 120 * 20, dtype=np.float32).reshape(2, 120, 20)
    flat = flatten_vitals(values)
    assert flat.shape == (2, 2400)
    assert flat[0, 0] == 0
    for t, c in ((0, 19), (15, 8), (119, 19)):
        np.testing.assert_array_equal(flat[:, t * 20 + c], values[:, t, c])
    with pytest.raises(ValueError, match="120,20"):
        flatten_vitals(np.zeros((2, 120, 14)))


def test_xgboost_cache_excludes_invalid_rows_and_keeps_order(workspace):
    from clinical_multimodal.pipelines.xgboost import cache_features
    root = make_fixed(workspace / "data")
    bundle = FixedCohortBundle(root, vital_only=True)
    processor = VitalNumericPreprocessor.fit(bundle, bundle.rows("train"))
    a = bundle.shard(0)
    a["vital_features"][0] = 0
    a["vital_features"][0, :, 4:8] = 1
    a["vital_observed_mask"][0] = False
    paths, y, rows = cache_features(bundle, processor, np.array([1, 0]), 6, workspace / "cache", 1)
    assert rows.tolist() == [1] and y.tolist() == [1]
    expected = flatten_vitals(processor.transform(bundle, [1])["vitals"])
    np.testing.assert_array_equal(np.load(paths[0][0]), expected)


@pytest.mark.skipif(importlib.util.find_spec("xgboost") is None, reason="XGBoost dependency not installed")
def test_xgboost_external_memory_training_reload_and_invalid_inputs(workspace):
    root = make_fixed(workspace / "data")
    remove_excluded_inputs(root)
    cfg = configuration(workspace, "xgboost")
    artifact = train_pipeline(cfg)
    predictor = Predictor(artifact)
    a = predictor.predict(root, rows=[5, 0], batch_size=1)
    b = Predictor(artifact).predict(root, rows=[5, 0], batch_size=2)
    np.testing.assert_array_equal([r["risk"] for r in a], [r["risk"] for r in b])
    assert predictor.model.num_features() == 2400
    assert (artifact / "model.ubj").is_file()
    assert not (artifact / "weights.pt").exists()
    assert evaluate_pipeline(artifact, root, workspace / "evaluation.json")["metrics"]["n"] == 2
    predictions = pd.read_csv(artifact.parent / "test_predictions.csv")
    np.testing.assert_allclose(predictions.risk, [r["risk"] for r in predictor.predict(root, rows=[4, 5])])
    bundle = FixedCohortBundle(root, vital_only=True)
    raw = bundle.shard(0)
    raw["vital_features"][:] = 0
    raw["vital_features"][:, :, 4:8] = 1
    raw["vital_observed_mask"][:] = False
    assert predictor.predict(bundle, rows=[0])[0]["risk"] is None
    with (artifact / "model.ubj").open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        load_artifact(artifact)


def test_cli_vital_only_defaults_and_explicit_models(workspace):
    from clinical_multimodal.runner import main
    cfg = configuration(workspace)
    cfg.model.vital_only = False
    cfg.training.models = ["news2"]
    path = workspace / "config.yaml"
    path.write_text(yaml.safe_dump(cfg.to_dict()), encoding="utf-8")
    with patch("clinical_multimodal.pipelines.train.train_pipeline") as train:
        main(["--config", str(path), "--vital_only"])
        assert [c.args[0].model.vital_encoder for c in train.call_args_list] == ["tcn", "lstm", "gru", "grud"]
        assert all(c.args[0].model.vital_only for c in train.call_args_list)
    with patch("clinical_multimodal.pipelines.train.train_pipeline") as train:
        main(["--config", str(path), "--vital_only", "--models", "xgboost"])
        assert train.call_args.args[0].model.vital_encoder == "xgboost"
    with pytest.raises(ValueError, match="requires"):
        main(["--config", str(path), "--models", "xgboost"])
    cfg.training.models = ["xgboost"]
    path.write_text(yaml.safe_dump(cfg.to_dict()), encoding="utf-8")
    with patch("clinical_multimodal.pipelines.train.train_pipeline") as train:
        main(["--config", str(path), "--vital_only", "--models", "xgboost"])
        assert train.call_args.args[0].model.vital_only


def test_vital_only_independent_cli_and_unlabelled_prediction(workspace):
    root = make_fixed(workspace / "data")
    remove_excluded_inputs(root)
    cfg = configuration(workspace, "gru")
    cfg.model.vital_only = False
    path = workspace / "config.yaml"
    path.write_text(yaml.safe_dump(cfg.to_dict()), encoding="utf-8")
    project = Path(__file__).resolve().parents[1]
    run = subprocess.run([sys.executable, "-B", str(project / "main.py"), "--config", str(path),
                          "--vital_only", "--models", "gru"], capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr
    artifact = next((workspace / "runs" / "vital_only" / "gru").glob("*/artifact"))
    frame = pd.read_csv(root / "sample_index.csv.gz")
    frame.drop(columns=[f"label_{h}h" for h in (1, 2, 3, 6)]).to_csv(root / "sample_index.csv.gz", index=False)
    assert len(Predictor(artifact, device="cpu").predict(root, rows=[0, 1])) == 2
