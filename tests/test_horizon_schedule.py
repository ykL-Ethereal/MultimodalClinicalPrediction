from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest

from clinical_multimodal.config import Config, load_config
from clinical_multimodal.runner import horizon_configs, main


def schedule_config():
    cfg = Config()
    cfg.data.horizons = [6, 3, 2, 1]
    cfg.training.models = ["gru", "tcn"]
    return cfg


def test_ordered_runs_are_independent_single_horizon_configs():
    cfg = schedule_config()
    runs = list(horizon_configs(cfg))
    assert [r.data.horizon for r in runs] == [6, 3, 2, 1]
    assert all(r.data.horizons == [] for r in runs)
    runs[0].training.models.clear()
    assert runs[1].training.models == cfg.training.models == ["gru", "tcn"]
    assert cfg.data.horizons == [6, 3, 2, 1]
    cfg.data.horizons = []
    cfg.data.horizon = 2
    assert [r.data.horizon for r in horizon_configs(cfg)] == [2]


def test_cli_completes_all_models_before_advancing_horizon():
    calls = []
    def train(cfg):
        calls.append((cfg.data.horizon, cfg.model.vital_encoder, cfg.data.horizons.copy()))
        return "mock-result"
    with patch("clinical_multimodal.runner.load_config", return_value=schedule_config()), \
         patch("clinical_multimodal.pipelines.train.train_pipeline", side_effect=train):
        main([])
    assert calls == [(h, model, []) for h in (6, 3, 2, 1) for model in ("gru", "tcn")]


def test_cli_explicit_horizon_overrides_schedule():
    calls = []
    with patch("clinical_multimodal.runner.load_config", return_value=schedule_config()), \
         patch("clinical_multimodal.pipelines.train.train_pipeline",
               side_effect=lambda cfg: calls.append(deepcopy(cfg))):
        main(["--horizon", "3"])
    assert [(c.data.horizon, c.model.vital_encoder) for c in calls] == [(3, "gru"), (3, "tcn")]
    assert all(c.data.horizons == [] for c in calls)


def test_audits_follow_the_same_schedule():
    calls = []
    with patch("clinical_multimodal.runner.load_config", return_value=schedule_config()), \
         patch("clinical_multimodal.pipelines.threshold.audit_threshold_splits",
               side_effect=lambda cfg: calls.append(cfg.data.horizon)):
        main(["--audit-threshold-splits"])
    assert calls == [6, 3, 2, 1]


def test_failed_run_stops_the_remaining_schedule():
    with patch("clinical_multimodal.runner.load_config", return_value=schedule_config()), \
         patch("clinical_multimodal.pipelines.train.train_pipeline", side_effect=RuntimeError("failed")) as train:
        with pytest.raises(RuntimeError, match="failed"):
            main([])
    assert train.call_count == 1


@pytest.mark.parametrize("horizons", [[6, 6], [4], [True], ["6"], "6,3,2,1", None])
def test_invalid_schedules_are_rejected(horizons):
    with pytest.raises(ValueError):
        Config.from_dict({"data": {"horizons": horizons}})


def test_default_research_configs_use_requested_order():
    root = Path(__file__).resolve().parents[1]
    for name in ("patient_cv_30min.yaml", "patient_cv_30min_uniform.yaml"):
        assert load_config(root / "configs" / name).data.horizons == [6, 3, 2, 1]


def test_horizon_sampling_budget_is_recorded_and_json_roundtrips():
    cfg = schedule_config()
    cfg.training.positive_fraction_by_horizon = {1: .03125}
    import json
    cfg = Config.from_dict(json.loads(json.dumps(cfg.to_dict())))
    runs = list(horizon_configs(cfg))
    assert [r.training.positive_fraction for r in runs] == [.0625, .0625, .0625, .03125]
    assert all(r.training.positive_fraction_by_horizon == {} for r in runs)
    assert cfg.training.positive_fraction_by_horizon == {1: .03125}
