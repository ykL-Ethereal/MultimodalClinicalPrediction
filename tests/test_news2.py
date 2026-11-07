"""User-specified inclusive bands, partial normalization and CLI integration."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import yaml

from clinical_multimodal.config import Config, load_config
from clinical_multimodal.data.fixed_adapter import FixedCohortBundle, VITAL_ORDER
from clinical_multimodal.inference.news2 import NEWS2Predictor
from clinical_multimodal.io import read_json, write_json
from clinical_multimodal.models.news2 import NEWS2Model, load_rules
from clinical_multimodal.pipelines.evaluate import evaluate_pipeline
from clinical_multimodal.pipelines.news2 import news2_pipeline, score_metrics
from tests.fixed_fixture import make_fixed


PROJECT = Path(__file__).resolve().parents[1]
RULES = PROJECT / "configs" / "news2_rules.yaml"


def normal_vitals(n=1):
    v = np.zeros((n, 120, len(VITAL_ORDER)), dtype=np.float64)
    v[:, :, :4] = [75, 75, 16, 98]
    v[:, :, 8], v[:, :, 10] = 120, 1
    return v


@pytest.mark.parametrize("parameter,channel,values,expected", [
    ("heart_rate", 0, [40,41,50,51,90,91,110,111,130,131], [3,1,1,0,0,1,1,2,2,3]),
    ("respiratory_rate", 2, [8,9,11,12,20,21,24,25], [3,1,1,0,0,2,2,3]),
    ("oxygen_saturation", 3, [91,92,93,94,95,96], [3,2,2,1,1,0]),
    ("systolic_blood_pressure", 8, [90,91,100,101,110,111,219,220], [3,2,2,1,1,0,0,3]),
])
def test_figure_one_boundaries(parameter, channel, values, expected):
    v = normal_vitals(len(values))
    v[:, :, channel] = np.array(values)[:, None]
    output = NEWS2Model(load_rules(RULES)).score(v, VITAL_ORDER)
    np.testing.assert_array_equal(output["components"][parameter]["scores"], expected)
    np.testing.assert_array_equal(output["scores"], expected)
    assert output["valid"].all()


def test_latest_observation_masks_historical_pressure_and_hr_only():
    v = normal_vitals()
    v[:, :, 4:8] = 1
    v[0, 20, :4], v[0, 20, 4:8] = [110, 200, 24, 93], 0
    v[0, -1, :4] = 999  # Unobserved fillers cannot change the selected readings.
    v[:, :, 10] = 0
    v[0, -1, 8], v[0, -1, 10] = 100, 1  # Last-value BP, no actual BP observation needed.
    output = NEWS2Model(load_rules(RULES)).score(v, VITAL_ORDER)
    assert output["scores"].tolist() == [7]
    assert output["components"]["heart_rate"]["sources"].tolist() == ["hr"]
    assert output["components"]["heart_rate"]["time_indices"].tolist() == [20]
    v[:, :, 4] = 1
    partial = NEWS2Model(load_rules(RULES)).score(v, VITAL_ORDER)
    assert partial["components"]["heart_rate"]["sources"].tolist() == [""]
    assert partial["scores"].tolist() == [6]  # PR cannot substitute for missing HR.
    assert partial["max_scores"].tolist() == [9]
    np.testing.assert_allclose(partial["normalized_scores"], [6 / 9])


def test_missing_values_are_not_normal_scores_and_fractional_bands_are_explicit():
    v = normal_vitals(2)
    v[0, :, 6] = 1
    v[1, :, 3] = np.nan
    output = NEWS2Model(load_rules(RULES)).score(v, VITAL_ORDER)
    assert output["valid"].all()
    assert output["max_scores"].tolist() == [9, 9]
    assert output["normalized_scores"].tolist() == [0, 0]
    v = normal_vitals()
    v[:, :, 2] = 20.9
    assert NEWS2Model(load_rules(RULES)).score(v, VITAL_ORDER)["scores"].tolist() == [2]


@pytest.mark.parametrize("available", range(16))
def test_user_example_and_every_missing_variable_combination(available):
    v = normal_vitals()
    v[:, :, :4] = [135, 75, 22, 90]
    v[:, :, 8] = 105
    expected_raw = 0
    for i, (mask_channel, present_value, points) in enumerate([(4,0,3), (6,0,2), (7,0,3), (10,1,1)]):
        present = bool(available & (1 << i))
        v[:, :, mask_channel] = present_value if present else 1 - present_value
        if present:
            expected_raw += points
    output = NEWS2Model(load_rules(RULES)).score(v, VITAL_ORDER)
    maximum = 3 * available.bit_count()
    assert output["scores"].tolist() == [expected_raw]
    assert output["max_scores"].tolist() == [maximum]
    assert output["valid"].tolist() == [bool(maximum)]
    if maximum:
        assert output["normalized_scores"][0] == pytest.approx(expected_raw / maximum)
    else:
        assert np.isnan(output["normalized_scores"][0])


@pytest.mark.parametrize("channel,upper,lower_score,next_score", [
    (0,40,3,1), (0,50,1,0), (0,90,0,1), (0,110,1,2), (0,130,2,3),
    (2,8,3,1), (2,11,1,0), (2,20,0,2), (2,24,2,3),
    (3,91,3,2), (3,93,2,1), (3,95,1,0),
    (8,90,3,2), (8,100,2,1), (8,110,1,0), (8,219,0,3),
])
def test_fractional_measurements_follow_inclusive_upper_limits(channel, upper, lower_score, next_score):
    v = normal_vitals(3)
    v[:, :, channel] = np.array([upper - .1, upper, upper + .1])[:, None]
    result = NEWS2Model(load_rules(RULES)).score(v, VITAL_ORDER)
    assert result["scores"].tolist() == [lower_score, lower_score, next_score]


def test_rule_changes_apply_without_code_changes(tmp_path):
    rules = load_rules(RULES)
    rules["parameters"]["heart_rate"]["cutpoints"][2] = 70
    rules["parameters"]["heart_rate"]["scores"][3] = 4
    rules["parameters"]["heart_rate"]["max_score"] = 4
    rules["alarm_threshold"] = 4
    path = tmp_path / "custom.yaml"
    path.write_text(yaml.safe_dump(rules), encoding="utf-8")
    assert NEWS2Model(load_rules(path)).score(normal_vitals(), VITAL_ORDER)["scores"].tolist() == [4]
    # Removing a parameter is controlled entirely by configuration, not the scorer.
    del rules["parameters"]["heart_rate"]
    rules["alarm_threshold"] = 2
    assert NEWS2Model(rules).score(normal_vitals(), VITAL_ORDER)["scores"].tolist() == [0]


def test_previous_artifact_rule_schema_keeps_original_semantics():
    rules = load_rules(RULES)
    rules.pop("interval_closed")
    rules.pop("normalization")
    rules["missing_policy"] = "require_all"
    cuts = {"heart_rate": [41,51,91,111,131], "respiratory_rate": [9,12,21,25],
            "oxygen_saturation": [92,94,96], "systolic_blood_pressure": [91,101,111,220]}
    for name, spec in rules["parameters"].items():
        spec["cutpoints"] = cuts[name]
        spec.pop("max_score")
    v = normal_vitals(2)
    v[:, :, 2] = 20.9
    v[1, :, 7] = 1
    output = NEWS2Model(rules).score(v, VITAL_ORDER)
    assert output["scores"].tolist() == [0, 0]
    assert output["valid"].tolist() == [True, False]


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(alarm_threshold=float("nan")),
    lambda r: r.update(alarm_threshold=99),
    lambda r: r.update(missing_policy="impute_zero"),
    lambda r: r.update(interval_closed="typo"),
    lambda r: r.update(normalization="probability"),
    lambda r: r["parameters"]["heart_rate"].update(cutpoints=[51,41]),
    lambda r: r["parameters"]["heart_rate"].update(scores=[0]),
    lambda r: r["parameters"]["heart_rate"].update(scores=[True,1,0,1,2,3]),
    lambda r: r["parameters"]["heart_rate"].update(max_score=2),
    lambda r: r["parameters"]["heart_rate"].update(max_score=0),
    lambda r: r.update(threshold_typo=2),
])
def test_invalid_rules_fail_fast(mutation):
    rules = load_rules(RULES)
    mutation(rules)
    with pytest.raises(ValueError):
        NEWS2Model(rules)


@pytest.fixture
def run(tmp_path):
    root = make_fixed(tmp_path / "data")
    cfg = Config()
    cfg.data.root = str(root)
    cfg.training.output_dir = str(tmp_path / "runs")
    cfg.training.models = ["news2"]
    cfg.model.vital_encoder = "news2"
    cfg.news2.batch_size = 2
    artifact = news2_pipeline(cfg)
    return cfg, artifact, FixedCohortBundle(root)


def test_artifact_replay_order_duplicates_alarm_and_no_label_leakage(run):
    cfg, artifact, bundle = run
    predictor = NEWS2Predictor(artifact)
    a = predictor.predict(bundle, rows=[5,0,5,2], batch_size=2)
    assert a == NEWS2Predictor(artifact).predict(bundle, rows=[5,0,5,2], batch_size=1)
    assert [p["sample_id"] for p in a] == ["5","0","5","2"]
    assert all(p["risk"] is None and p["variant"] == "reduced" for p in a)
    assert all(p["alarm"] == (p["score"] >= 2) for p in a)
    assert all(p["raw_score"] == p["score"] for p in a)
    assert all(p["max_score"] == 12 for p in a)
    assert all(p["normalized_score"] == p["raw_score"] / p["max_score"] for p in a)
    bundle.targets = {}
    assert predictor.predict(bundle, rows=[5,0,5,2]) == a
    assert predictor.predict(bundle, rows=[]) == []
    with pytest.raises(ValueError):
        predictor.predict(bundle, rows=[.5])
    cfg.news2.rules_file = "nonexistent.yaml"  # Artifact uses its own rule snapshot.
    assert NEWS2Predictor(artifact).predict(bundle, rows=[5])[0] == a[0]


def test_pipeline_outputs_and_shared_evaluation_entry(run, tmp_path):
    cfg, artifact, bundle = run
    output = artifact.parent
    assert read_json(output / "status.json")["status"] == "completed"
    assert read_json(output / "run_summary.json")["fitted"] is False
    assert not (output / "best.pt").exists()
    stored = read_json(output / "test_report.json")
    repeated = evaluate_pipeline(artifact, cfg.data.root, tmp_path / "evaluated.json")
    assert stored == repeated
    assert stored["metrics"]["brier"] is None
    assert stored["metrics"]["log_loss"] is None
    assert stored["normalized_metrics"]["auroc"] is not None
    assert stored["normalized_metrics"]["threshold"] is None
    predictions = [json.loads(s) for s in (output / "test_predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(predictions) == 2
    assert predictions[0]["omitted_parameters"] == ["temperature", "consciousness", "supplemental_oxygen"]


def test_missing_retained_parameter_is_excluded_from_score_and_maximum(run, tmp_path):
    _, artifact, bundle = run
    shard = bundle.shard(0)
    shard["vital_features"][:, :, 6] = 1
    result = NEWS2Predictor(artifact).predict(bundle, rows=[0])[0]
    assert result["score"] == result["raw_score"] == 4
    assert result["max_score"] == 9
    assert result["normalized_score"] == pytest.approx(4 / 9)
    assert result["alarm"] is True
    assert result["status"] == "partial_input"
    assert result["missing_parameters"] == ["respiratory_rate"]
    assert result["components"]["respiratory_rate"]["score"] is None
    from clinical_multimodal.pipelines.news2 import evaluate_news2
    report = evaluate_news2(NEWS2Predictor(artifact), bundle, tmp_path / "partial_report.json", split="train")
    assert report["invalid_inputs"] == 0
    assert report["partial_inputs"] == 2
    assert report["metrics"]["n"] == 2


def test_all_missing_returns_null_scores_and_zero_maximum(run):
    _, artifact, bundle = run
    shard = bundle.shard(0)
    shard["vital_features"][:, :, 4:8] = 1
    shard["vital_features"][:, :, 10] = 0
    result = NEWS2Predictor(artifact).predict(bundle, rows=[0])[0]
    assert result["raw_score"] is result["normalized_score"] is result["score"] is result["alarm"] is None
    assert result["max_score"] == 0
    assert result["risk"] is None


def test_normalized_evaluation_uses_each_samples_available_maximum(run, tmp_path):
    _, artifact, bundle = run
    from clinical_multimodal.pipelines.news2 import evaluate_news2
    shard = bundle.shard(0)
    v = shard["vital_features"]
    v[:] = normal_vitals(2)
    # Negative sample: only SpO2=93, raw=2, maximum=3, normalized=2/3.
    v[0, :, [4, 5, 6]] = 1
    v[0, :, 10] = 0
    v[0, :, 3] = 93
    # Positive sample: four variables, HR=91 and SBP=100, raw=3, normalized=3/12.
    v[1, :, 0], v[1, :, 8] = 91, 100
    report = evaluate_news2(NEWS2Predictor(artifact), bundle, tmp_path / "ranking.json", split="train")
    assert report["metrics"]["auroc"] == 1
    assert report["normalized_metrics"]["auroc"] == 0
    assert report["partial_inputs"] == 1


@pytest.mark.parametrize("target", ["news2_rules.yaml", "manifest.json"])
def test_artifact_integrity(run, target):
    _, artifact, _ = run
    path = artifact / target
    if target.endswith("yaml"):
        path.write_text(path.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
    else:
        manifest = read_json(path)
        manifest["horizon_hours"] = 1
        write_json(path, manifest)
    with pytest.raises(ValueError, match="integrity"):
        NEWS2Predictor(artifact)


def test_score_metrics_include_equality_and_degenerate_cases():
    result = score_metrics([0,0,1,1], [0,1,2,12], 2)
    assert result["auroc"] == result["auprc"] == result["sensitivity"] == result["specificity"] == 1
    assert score_metrics([], [], 2)["status"] == "no_valid_labels"
    assert score_metrics([0], [0], 2)["auroc"] is None


def test_cli_does_not_import_neural_dependencies_and_resolves_rules(run, tmp_path):
    cfg, _, _ = run
    cfg.news2.rules_file = "rules.yaml"
    (tmp_path / "rules.yaml").write_text(RULES.read_text(encoding="utf-8"), encoding="utf-8")
    config_path = tmp_path / "train.yaml"
    config_path.write_text(yaml.safe_dump(cfg.to_dict()), encoding="utf-8")
    assert load_config(config_path).news2.rules_file == str(tmp_path / "rules.yaml")
    script = '''
import sys
class DenyNeural:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers'}:
            raise RuntimeError('NEWS2 must not import ' + fullname)
sys.meta_path.insert(0, DenyNeural())
from clinical_multimodal.runner import main
main(['--config', sys.argv[1], '--models', 'news2'])
'''
    result = subprocess.run([sys.executable, "-B", "-c", script, str(config_path)],
        env=dict(os.environ, PYTHONPATH=str(PROJECT / "src")), capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert "no fitting" in result.stdout
