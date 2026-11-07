"""Decision boundaries, leakage isolation and saved decisions across all five models."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from clinical_multimodal.config import Config, load_config
from clinical_multimodal.data.fixed_adapter import FixedCohortBundle
from clinical_multimodal.evaluation.calibration import sigmoid, PlattCalibrator
from clinical_multimodal.evaluation.thresholds import select_threshold, apply_threshold, apply_policy
from clinical_multimodal.inference.predictor import Predictor
from clinical_multimodal.io import read_json, write_json, fingerprint
from clinical_multimodal.pipelines.evaluate import evaluate_pipeline
from clinical_multimodal.pipelines.threshold import validation_partition, fit_threshold
from clinical_multimodal.pipelines.train import train_pipeline
from tests.fixed_fixture import make_fixed
from tests.helpers import TestWorkspace
from tests.test_vital_only import configuration


def test_recall_constraint_optimizes_precision_and_breaks_ties_high():
    # At .9 and .7 precision=1; both satisfy recall=.5, so choose .9.
    result = select_threshold([1, 1, 0, 0], [.9, .7, .6, .1], target_recall=.5)
    assert result["threshold"] == .9
    assert result["selection_metrics"]["recall"] == .5
    assert select_threshold([1, 1, 0, 0], [.9, .7, .6, .1], target_recall=.8)["threshold"] == .7
    # Lowering a threshold can improve precision, so the first feasible boundary is insufficient.
    assert select_threshold([0, 1, 1, 0], [.9, .8, .7, .1], target_recall=.5)["threshold"] == .7


def test_max_f1_ties_duplicates_endpoints_and_inclusive_rule():
    # F1(.9)=F1(.7)=2/3, higher boundary wins.
    assert select_threshold([1, 0, 0, 1], [.9, .8, .8, .7], strategy="max_f1")["threshold"] == .9
    result = select_threshold([0, 1, 1], [.7, .7, .3], target_recall=1.)
    assert result["threshold"] == .3 and result["selection_metrics"]["recall"] == 1.
    np.testing.assert_array_equal(apply_threshold([0, .5, 1], .5), [0, 1, 1])
    assert select_threshold([0, 1], [0, 0])["threshold"] == 0
    assert select_threshold([0, 1], [0, 1])["threshold"] == 1
    assert select_threshold([0, 1], [.1, .8], strategy="fixed", threshold=.42)["threshold"] == .42


@pytest.mark.parametrize("labels,probs,kwargs", [
    ([], [], {}), ([1, 1], [.1, .2], {}), ([0, 2], [.1, .2], {}),
    ([0, 1], [np.nan, .2], {}), ([0, 1], [-.1, .2], {}),
    ([0], [.1, .2], {}), ([0, 1], [[.1, .2]], {}),
    ([0, 1], [.1, .2], {"target_recall": 0}),
    ([0, 1], [.1, .2], {"target_recall": 1.1}),
    ([0, 1], [.1, .2], {"strategy": "fixed"}),
    ([0, 1], [.1, .2], {"strategy": "unknown"}),
])
def test_invalid_selection_inputs(labels, probs, kwargs):
    with pytest.raises(ValueError):
        select_threshold(labels, probs, **kwargs)


@pytest.mark.parametrize("threshold", [None, np.nan, np.inf, -.1, 1.1, True, "0.5"])
def test_invalid_saved_boundary(threshold):
    with pytest.raises(ValueError, match="threshold"):
        apply_threshold([.5], threshold)


def test_policy_space_horizon_and_missing_threshold_errors():
    policy = {**select_threshold([0, 1], [.1, .8]), "probability_space": "calibrated", "horizon_hours": 1}
    assert apply_policy([.8], policy, probability_space="calibrated", horizon_hours=1).tolist() == [1]
    for changed in ({"probability_space": "raw"}, {"horizon_hours": 2}, {"threshold": None}):
        with pytest.raises(ValueError):
            apply_policy([.8], {**policy, **changed}, probability_space="calibrated", horizon_hours=1)
    with pytest.raises(ValueError, match="Missing"):
        apply_policy([.8], None, probability_space="calibrated", horizon_hours=1)


def collected_for(bundle, rows, logits=None):
    y, mask = bundle.labels(rows, 6)
    return {"targets": y, "label_mask": mask, "valid": np.ones(len(rows), bool),
            "logits": np.array([-1., 1.] * (len(rows) // 2)) if logits is None else logits,
            "metadata": bundle.metadata.iloc[rows][["sample_id"]].to_dict("records")}


def test_partition_and_fitting_exclude_test_and_calibrator_fit_patients():
    with TestWorkspace() as directory:
        root = make_fixed(Path(directory) / "data", validation_patients=9)
        bundle = FixedCohortBundle(root, vital_only=True)
        cfg = configuration(Path(directory))
        cfg.training.threshold_strategy, cfg.training.threshold = "recall_constraint", None
        cfg.training.calibration = True
        parts = validation_partition(bundle, cfg)
        for i, a in enumerate(parts):
            assert len(a)
            for b in parts[i+1:]:
                assert set(bundle.metadata.iloc[a].patient_id).isdisjoint(bundle.metadata.iloc[b].patient_id)
        assert sorted(np.concatenate(parts)) == bundle.rows("validation").tolist()
        np.testing.assert_equal(parts, validation_partition(bundle, cfg))
        rows = parts[2]
        cal = PlattCalibrator(.2, -3.)
        policy = fit_threshold(bundle, cfg, rows, collected_for(bundle, rows), cal, parts)
        assert policy["threshold"] in cal.predict(collected_for(bundle, rows)["logits"])
        assert policy["threshold"] not in sigmoid(collected_for(bundle, rows)["logits"])
        assert policy["source"]["independent_of_model_and_calibrator_fit"]
        for forbidden in (bundle.rows("test"), bundle.rows("train"), parts[0], parts[1]):
            with pytest.raises(ValueError, match="validation partition"):
                fit_threshold(bundle, cfg, forbidden, collected_for(bundle, forbidden), cal, parts)
        # Test label changes cannot influence partitioning or threshold selection.
        bundle.targets["y"][bundle.rows("test")] = 1
        assert fit_threshold(bundle, cfg, rows, collected_for(bundle, rows), cal, parts) == policy
        bad = collected_for(bundle, rows)
        bad["metadata"] = list(reversed(bad["metadata"]))
        with pytest.raises(ValueError, match="identities"):
            fit_threshold(bundle, cfg, rows, bad, cal, parts)


@pytest.mark.parametrize("encoder", ["tcn", "lstm", "gru", "grud", "xgboost"])
def test_calibrated_training_saved_decision_and_evaluation(encoder):
    if encoder == "xgboost":
        pytest.importorskip("xgboost")
    with TestWorkspace() as directory:
        workspace = Path(directory)
        root = make_fixed(workspace / "data", validation_patients=3)
        cfg = configuration(workspace, encoder)
        cfg.training.threshold_strategy, cfg.training.threshold = "recall_constraint", None
        cfg.training.calibration = True
        # Spy on both fitting stages: every input must be held-out, never test.
        with patch("clinical_multimodal.pipelines.train.fit_threshold", wraps=fit_threshold) as neural_fit, \
             patch("clinical_multimodal.pipelines.xgboost.fit_threshold", wraps=fit_threshold) as tree_fit:
            artifact = train_pipeline(cfg)
        call = (tree_fit if encoder == "xgboost" else neural_fit).call_args
        bundle, _, rows, collected, cal, parts = call.args
        assert bundle.metadata.iloc[rows].split.eq("validation").all()
        assert np.array_equal(rows, parts[2]) and cal is not None
        policy = read_json(artifact / "threshold.json")
        assert policy == read_json(artifact.parent / "threshold.json")
        assert policy["selection_metrics"]["n"] == 2 and policy["probability_space"] == "calibrated"
        predictor = Predictor(artifact, device="cpu")
        with patch("clinical_multimodal.evaluation.thresholds.select_threshold", side_effect=AssertionError("inference fitted")), \
             patch.object(PlattCalibrator, "fit", side_effect=AssertionError("inference calibrated")):
            predictions = predictor.predict(root, rows=bundle.rows("test"))
            report = evaluate_pipeline(artifact, root, workspace / "evaluation.json", device="cpu")
        p = [x["risk"] for x in predictions]
        assert [x["prediction"] for x in predictions] == apply_policy(p, policy,
            probability_space="calibrated", horizon_hours=6).tolist()
        assert [x["alarm"] for x in predictions] == [x >= policy["threshold"] for x in p]
        saved = read_json(artifact.parent / "test_report.json")
        for key in ("precision", "recall", "f1", "specificity", "confusion_matrix", "auroc", "auprc", "brier", "log_loss"):
            assert key in saved["metrics"]
            if key == "confusion_matrix":
                assert saved["metrics"][key] == report["metrics"][key]
            else:
                assert saved["metrics"][key] == pytest.approx(report["metrics"][key], abs=1e-8)
        assert saved["fixed_0_5"]["threshold"] == .5
        assert saved["continuous_warning_metrics"]["missing_fields"] == ["event_time", "monitor_end", "monitor_start"]
        for bad_rows in ([.5], [-1], [100], [[0]], [True]):
            with pytest.raises(ValueError, match="row indices"):
                predictor.predict(root, rows=bad_rows)
        # Re-sign deliberate semantic corruption, so failure tests more than the checksum.
        manifest = read_json(artifact / "manifest.json")
        for key, value, message in (("threshold", None, "threshold"),
                                    ("probability_space", "raw", "space mismatch"),
                                    ("horizon_hours", 1, "horizon mismatch")):
            altered = deepcopy(manifest)
            altered["threshold_policy"][key] = value
            altered.pop("model_version")
            altered["model_version"] = fingerprint(altered)
            write_json(artifact / "manifest.json", altered)
            with pytest.raises(ValueError, match=message):
                Predictor(artifact, device="cpu")


def test_research_configuration_and_horizons():
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs/vital_thresholds.yaml")
    assert cfg.model.vital_only and cfg.training.calibration
    assert set(cfg.training.models) == {"tcn", "lstm", "gru", "grud", "xgboost"}
    assert cfg.training.target_recall == .8
    for horizon in (1, 2, 3, 6):
        cfg.data.horizon = horizon
        cfg.validate()
    for raw in ({"threshold_strategy": "other"}, {"target_recall": 0},
                {"threshold_fraction": 1}, {"threshold_strategy": "fixed"},
                {"threshold": .195414}):
        with pytest.raises(ValueError):
            Config.from_dict({"training": raw})
