from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from clinical_multimodal.config import Config, load_config
from clinical_multimodal.data.balanced_sampler import PatientTimeBatchSampler
from clinical_multimodal.data.fixed_adapter import FixedCohortBundle
from clinical_multimodal.data.resample import resample_cohort, spaced_rows
from clinical_multimodal.data.vital_cache import CachedVitalDataset
from clinical_multimodal.evaluation.bootstrap import patient_bootstrap
from clinical_multimodal.evaluation.thresholds import select_threshold
from clinical_multimodal.inference.predictor import Predictor
from clinical_multimodal.io import read_json, write_json
from clinical_multimodal.pipelines.patient_cv import cv_pipeline, finalize_cv, patient_folds
from clinical_multimodal.preprocessing.vital_numeric import VitalNumericPreprocessor
from tests.fixed_fixture import make_fixed
from tests.helpers import TestWorkspace


def research_fixture(directory):
    root = make_fixed(Path(directory) / "data", validation_patients=5)
    frame = pd.read_csv(root / "sample_index.csv.gz")
    frame.loc[:7, "split"] = "train"
    frame["prediction_time"] = ["2026-01-01 02:00:00", "2026-01-01 02:30:00"] * 7
    frame["time_to_event_minutes"] = [np.nan, 120.] * 7
    frame.to_csv(root / "sample_index.csv.gz", index=False)
    frame[["patient_group_id", "split"]].drop_duplicates().to_csv(root / "patient_splits.csv", index=False)
    for split in ("train", "validation", "test"):
        np.save(root / "splits" / f"{split}_sample_ids.npy", frame.loc[frame.split == split, "sample_id"].to_numpy())
    for path in sorted((root / "tensors").glob("*.npz")):
        with np.load(path) as z:
            arrays = {k: z[k] for k in z.files}
        arrays["prediction_second"][1] += 1800
        np.savez_compressed(path, **arrays)
    cfg = Config()
    cfg.data.root = str(root)
    cfg.model.vital_only = True
    cfg.model.vital_encoder = "gru"
    cfg.model.rnn_hidden = 4
    cfg.model.embedding_dim = 8
    cfg.model.dropout = 0
    t = cfg.training
    t.models = ["gru"]
    t.protocol, t.selection_metric, t.threshold_strategy = "patient_cv", "fpr_at_recall", "min_fp_recall"
    t.cv_folds, t.epochs, t.batch_size, t.steps_per_epoch = 2, 1, 4, 2
    t.sampling, t.positive_fraction = "patient_time_balanced", .25
    t.use_numeric_cache, t.device, t.precision = True, "cpu", "float32"
    t.numeric_cache_dir = str(Path(directory) / "cache")
    t.output_dir = str(Path(directory) / "runs")
    t.num_threads, t.bootstrap_replicates = 1, 20
    return root, cfg.validate()


def test_min_fp_differs_from_max_precision_and_never_breaks_score_ties():
    y = np.array([1, 1, 1, 1, 0, 1])
    p = np.array([.9, .8, .7, .6, .6, .5])
    minimum = select_threshold(y, p, strategy="min_fp_recall", target_recall=.8)
    precision = select_threshold(y, p, strategy="recall_constraint", target_recall=.8)
    assert minimum["threshold"] == .6 and minimum["selection_metrics"]["fp"] == 1
    assert precision["threshold"] == .5
    assert minimum["selection_metrics"]["recall"] == .8


def test_resampling_is_label_blind_keeps_120_minutes_and_all_modalities():
    with TestWorkspace() as temp:
        root, _ = research_fixture(temp)
        frame = pd.read_csv(root / "sample_index.csv.gz")
        selected = spaced_rows(frame, 60)
        changed = frame.copy()
        changed["label_6h"] = 1 - frame.label_6h
        changed["time_to_event_minutes"] = 999
        np.testing.assert_array_equal(selected, spaced_rows(changed, 60))
        output = resample_cohort(root, Path(temp) / "resampled", 60)
        b = FixedCohortBundle(output)
        assert len(b.metadata) == 7 and b.schema["history_minutes"] == 120
        assert b.metadata.source_sample_id.tolist() == list(range(0, 14, 2))
        np.testing.assert_array_equal(b.get_numeric([0])["lab_features"], FixedCohortBundle(root).get_numeric([0])["lab_features"])
        with pytest.raises(ValueError, match="new output"):
            resample_cohort(root, output)


def test_patient_folds_isolate_patients_and_refit_statistics():
    with TestWorkspace() as temp:
        root, cfg = research_fixture(temp)
        b = FixedCohortBundle(root, vital_only=True)
        folds = patient_folds(b, 6, 2, 42)
        assert sorted(np.concatenate([v for _, v in folds]).tolist()) == b.rows("train").tolist()
        for train, val in folds:
            p = VitalNumericPreprocessor.fit(b, train)
            assert set(p.fit_provenance["patient_ids"]) == set(b.metadata.iloc[train].patient_id)
            assert not set(p.fit_provenance["patient_ids"]) & set(b.metadata.iloc[val].patient_id)
            expected = b.get_numeric(train)["vital_features"][:, :, 0].mean()
            assert p.vital_mean[0] == pytest.approx(expected)
        with pytest.raises(ValueError, match="Too few positive"):
            patient_folds(b, 6, 5, 42)


def test_balanced_sampling_caps_exposure_and_cache_preserves_inputs():
    with TestWorkspace() as temp:
        root, cfg = research_fixture(temp)
        b = FixedCohortBundle(root, vital_only=True)
        rows = b.rows("train")
        p = VitalNumericPreprocessor.fit(b, rows)
        ds = CachedVitalDataset(b, rows, p, 6, cfg.training.numeric_cache_dir)
        expected = p.transform(b, [3, 0])["vitals"]
        actual = np.stack([s["inputs"]["vitals"] for s in ds.__getitems__([3, 0])])
        np.testing.assert_array_equal(expected, actual)
        sampler = PatientTimeBatchSampler(ds, 4, 42, steps=4, positive_fraction=.5, max_positive_repeats=2)
        batches = list(sampler)
        for batch in batches:
            positives = [i for i in batch if ds.y[i] == 1]
            assert len(positives) == 2
            assert ds.metadata.iloc[positives].patient_id.nunique() == 2
        assert sampler.audit["max_positive_repeats"] <= 2
        with pytest.raises(ValueError, match="budget"):
            PatientTimeBatchSampler(ds, 4, 42, steps=5, positive_fraction=.5, max_positive_repeats=2)
        ds.close()


def test_cv_never_uses_validation_or_test_and_finalization_freezes_artifact_first():
    with TestWorkspace() as temp:
        root, cfg = research_fixture(temp)
        labels = FixedCohortBundle.labels
        def train_only(self, rows, horizon):
            assert self.metadata.iloc[rows].split.eq("train").all()
            return labels(self, rows, horizon)
        with patch.object(FixedCohortBundle, "labels", train_only):
            summary_path = cv_pipeline(cfg)
        summary = read_json(summary_path)
        assert not summary["test_evaluated"] and not summary["validation_used_for_selection"]
        assert not list(summary_path.parent.rglob("test_report.json"))
        final_root = Path(temp) / "final_runs"
        def frozen_before_test(self, rows, horizon):
            if self.metadata.iloc[rows].split.eq("test").any():
                assert list(final_root.rglob("artifact/threshold.json"))
            return labels(self, rows, horizon)
        with patch.object(FixedCohortBundle, "labels", frozen_before_test):
            artifact = finalize_cv(summary_path, final_root)
        policy = read_json(artifact / "threshold.json")
        assert policy["source"]["total_windows"] == 4
        assert policy["source"]["independent_of_model_and_calibrator_fit"]
        predictor = Predictor(artifact, device="cpu")
        result = predictor.predict(root, rows=[0, 1])
        assert all(x["risk"] is not None for x in result)
        report = read_json(artifact.parent / "test_report.json")
        assert report["patient_bootstrap"]["threshold_reselected"] is False
        with pytest.raises(FileExistsError):
            finalize_cv(summary_path, final_root)


def test_cv_rejects_changed_data_and_unsafe_configuration():
    with TestWorkspace() as temp:
        root, cfg = research_fixture(temp)
        summary_path = cv_pipeline(cfg)
        schema = read_json(root / "dataset_schema.json")
        schema["version"] += "-changed"
        write_json(root / "dataset_schema.json", schema)
        with pytest.raises(ValueError, match="Dataset changed"):
            finalize_cv(summary_path)
        for field, value in (("calibration", True), ("class_weight", True)):
            changed = deepcopy(cfg)
            setattr(changed.training, field, value)
            with pytest.raises(ValueError):
                changed.validate()


def test_patient_cluster_bootstrap_retains_frozen_rule():
    report = patient_bootstrap([0, 1, 0, 1], [.1, .9, .1, .9], ["a", "a", "b", "b"], .5, replicates=50)
    assert report["intervals"]["recall"]["lower"] == 1
    assert report["intervals"]["fpr"]["upper"] == 0
    assert report["patients"] == 2 and not report["threshold_reselected"]


def test_research_configs_share_data_protocol_and_keep_projection_dropout():
    root = Path(__file__).resolve().parents[1]
    balanced = load_config(root / "configs/patient_cv_30min.yaml")
    uniform = load_config(root / "configs/patient_cv_30min_uniform.yaml")
    assert uniform.training.sampling == "uniform_budget"
    assert balanced.data == uniform.data
    assert balanced.training.protocol == uniform.training.protocol == "patient_cv"
    # Epoch/patience budgets are user-editable; the comparison contract groups
    # differing budgets separately instead of requiring identical config files.
    assert balanced.training.target_recall == uniform.training.target_recall
    from clinical_multimodal.models.vital_rnn import VitalRNN
    rnn = VitalRNN(balanced.model, input_channels=20)
    assert rnn.rnn.dropout == 0.0 and rnn.project[0].p == .1
    historical = Config().model
    historical.vital_encoder = "gru"
    assert VitalRNN(historical, input_channels=20).rnn.dropout == historical.dropout
