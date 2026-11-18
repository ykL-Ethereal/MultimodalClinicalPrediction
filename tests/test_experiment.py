from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest

from clinical_multimodal.evaluation.cv_selection import compare_summaries
from clinical_multimodal.io import fingerprint, read_json, write_json
from clinical_multimodal.pipelines.experiment import experiment_pipeline
from clinical_multimodal.runner import horizon_configs, main
from tests.helpers import TestWorkspace
from tests.test_patient_cv import research_fixture


def fake_summary(cfg, fpr=.2, std=.01, ap=.1, identity="same"):
    folder = Path(cfg.training.output_dir) / "cv" / cfg.model.vital_encoder / "mock"
    folder.mkdir(parents=True)
    (folder / "best.pt").write_bytes(b"preserve intermediate checkpoint")
    summary = {"protocol": "train_patient_cv_v1", "config": cfg.to_dict(), "dataset_identity": identity,
               "code_sha256": "same_code", "partition_sha256": f"h{cfg.data.horizon}",
               "mean_fpr": fpr, "std_fpr": std, "mean_auprc": ap, "final_epochs": 2,
               "test_evaluated": False, "validation_used_for_selection": False}
    summary["sha256"] = fingerprint(summary)
    path = folder / "cv_summary.json"
    write_json(path, summary)
    return path


def test_chain_selects_one_model_per_horizon_and_keeps_all_candidates():
    with TestWorkspace() as temp:
        _, cfg = research_fixture(temp)
        cfg.data.horizons = [6, 3, 2, 1]
        cfg.training.models = ["gru", "lstm"]
        calls = []
        def cv(c):
            calls.append(("cv", c.data.horizon, c.model.vital_encoder))
            winner = "gru" if c.data.horizon in (6, 2) else "lstm"
            return fake_summary(c, fpr=.1 if c.model.vital_encoder == winner else .3)
        def finalize(path, root):
            selected = read_json(path)
            h, name = selected["config"]["data"]["horizon"], selected["config"]["model"]["vital_encoder"]
            record = read_json(root / "model_selection.json")
            assert record["winner"]["model"] == name and not record["test_used"]
            assert len(record["candidates"]) == 2
            calls.append(("final", h, name))
            artifact = root / "final" / name / "mock" / "artifact"
            artifact.mkdir(parents=True)
            return artifact
        with patch("clinical_multimodal.pipelines.experiment.cv_pipeline", side_effect=cv), \
             patch("clinical_multimodal.pipelines.experiment.finalize_cv", side_effect=finalize):
            state_path = experiment_pipeline(cfg, list(horizon_configs(cfg)))
        expected = []
        for h in (6, 3, 2, 1):
            expected += [("cv", h, "gru"), ("cv", h, "lstm"), ("final", h, "gru" if h in (6, 2) else "lstm")]
        assert calls == expected
        state = read_json(state_path)
        assert state["status"] == "completed"
        assert read_json(state_path.parent / "experiment_plan.json")["plans"][0]["training"]["models"] == ["gru", "lstm"]
        for item in state["horizons"]:
            assert item["status"] == "completed"
            for path in item["cv_summaries"]:
                assert Path(path).with_name("best.pt").read_bytes() == b"preserve intermediate checkpoint"


def test_failure_preserves_completed_cv_and_does_not_finalize_partial_candidates():
    with TestWorkspace() as temp:
        _, cfg = research_fixture(temp)
        cfg.training.models = ["gru", "lstm"]
        def cv(c):
            if c.model.vital_encoder == "lstm":
                raise RuntimeError("training failed")
            return fake_summary(c)
        with patch("clinical_multimodal.pipelines.experiment.cv_pipeline", side_effect=cv), \
             patch("clinical_multimodal.pipelines.experiment.finalize_cv") as finalize:
            with pytest.raises(RuntimeError, match="training failed"):
                experiment_pipeline(cfg, list(horizon_configs(cfg)))
        finalize.assert_not_called()
        state_path = next(Path(cfg.training.output_dir).rglob("experiment_state.json"))
        state = read_json(state_path)
        assert state["status"] == "failed"
        assert len(state["horizons"][0]["cv_summaries"]) == 1
        assert Path(state["horizons"][0]["cv_summaries"][0]).with_name("best.pt").exists()


def test_incompatible_candidate_data_refuses_finalization():
    with TestWorkspace() as temp:
        _, cfg = research_fixture(temp)
        cfg.training.models = ["gru", "lstm"]
        with patch("clinical_multimodal.pipelines.experiment.cv_pipeline",
                   side_effect=lambda c: fake_summary(c, identity=c.model.vital_encoder)), \
             patch("clinical_multimodal.pipelines.experiment.finalize_cv") as finalize:
            with pytest.raises(ValueError, match="differ"):
                experiment_pipeline(cfg, list(horizon_configs(cfg)))
        finalize.assert_not_called()


def test_selection_uses_fpr_then_sd_then_ap_and_rejects_holdout_selection():
    with TestWorkspace() as temp:
        _, cfg = research_fixture(temp)
        paths = []
        for name, fpr, std, ap in [("gru", .1, .02, .9), ("lstm", .1, .01, .3),
                                    ("tcn", .1, .01, .4), ("grud", .2, 0., 1.)]:
            c = deepcopy(cfg)
            c.model.vital_encoder = name
            paths.append(fake_summary(c, fpr, std, ap))
        ranking = next(iter(compare_summaries(paths).values()))["candidates"]
        assert [r["model"] for r in ranking] == ["tcn", "lstm", "gru", "grud"]
        s = read_json(paths[0])
        s["test_evaluated"] = True
        s["sha256"] = fingerprint({k: v for k, v in s.items() if k != "sha256"})
        write_json(paths[0], s)
        with pytest.raises(ValueError, match="Train-only"):
            compare_summaries(paths)


def test_default_cli_chains_and_cv_only_keeps_manual_workflow():
    with TestWorkspace() as temp:
        _, cfg = research_fixture(temp)
        cfg.data.horizons = [6, 3]
        with patch("clinical_multimodal.runner.load_config", return_value=deepcopy(cfg)), \
             patch("clinical_multimodal.pipelines.experiment.experiment_pipeline", return_value="state") as workflow:
            main([])
        assert [c.data.horizon for c in workflow.call_args.args[1]] == [6, 3]
        with patch("clinical_multimodal.runner.load_config", return_value=deepcopy(cfg)), \
             patch("clinical_multimodal.pipelines.experiment.experiment_pipeline") as workflow, \
             patch("clinical_multimodal.pipelines.train.train_pipeline", return_value="cv") as cv:
            main(["--cv-only"])
        workflow.assert_not_called()
        assert cv.call_count == 2


def test_real_tiny_chain_preserves_fold_predictions_weights_and_final_reports():
    with TestWorkspace() as temp:
        _, cfg = research_fixture(temp)
        cfg.training.models = ["gru", "lstm"]
        state_path = experiment_pipeline(cfg, list(horizon_configs(cfg)))
        state = read_json(state_path)
        assert state["status"] == "completed"
        item = state["horizons"][0]
        assert len(item["cv_summaries"]) == 2
        for summary in item["cv_summaries"]:
            for fold in ("fold_1", "fold_2"):
                for name in ("best.pt", "last.pt", "history.json", "preprocessing.json", "selection_report.json",
                             "selection_predictions.csv", "sampling_epoch_001.json", "exposure_epoch_001.npy"):
                    assert (Path(summary).parent / fold / name).is_file()
        artifact = Path(item["artifact"])
        assert artifact.is_dir()
        assert (artifact.parent / "validation_report.json").is_file()
        assert (artifact.parent / "test_report.json").is_file()
        assert len(list(state_path.parent.rglob("test_report.json"))) == 1
        assert read_json(artifact / "threshold.json")["source"]["model_selection"]["summary_sha256"] == \
               read_json(item["selection_record"])["winner"]["summary_sha256"]


def test_single_python_command_runs_two_horizons_through_test():
    import os
    import subprocess
    import sys
    import yaml
    with TestWorkspace() as temp:
        _, cfg = research_fixture(temp)
        cfg.data.horizons = [6, 3]
        path = Path(temp) / "automatic.yaml"
        path.write_text(yaml.safe_dump(cfg.to_dict(), allow_unicode=True), encoding="utf-8")
        project = Path(__file__).resolve().parents[1]
        result = subprocess.run([sys.executable, "-B", str(project / "main.py"), "--config", str(path)],
                                capture_output=True, text=True, timeout=90,
                                env=dict(os.environ, PYTHONPATH=str(project / "src")))
        assert result.returncode == 0, result.stdout + result.stderr
        state_path = next(Path(cfg.training.output_dir).rglob("experiment_state.json"))
        state = read_json(state_path)
        assert state["status"] == "completed"
        assert [h["horizon_hours"] for h in state["horizons"]] == [6, 3]
        assert len(list(state_path.parent.rglob("test_report.json"))) == 2
