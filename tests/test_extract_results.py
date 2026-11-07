"""Exercise result extraction without training dependencies or real cohort data."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location(
    "extract_results", Path(__file__).resolve().parents[1] / "extract_results.py")
extract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extract)


class ExtractResultsTests(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory()
        self.addCleanup(self.workspace.cleanup)
        self.root = Path(self.workspace.name)

    def write(self, directory, name, value):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / name).write_text(json.dumps(value), encoding="utf-8")

    def run_dir(self, model="tcn", stamp="20261102_010000_000000", status="completed"):
        directory = self.root / "runs" / "vital_only" / model / stamp
        self.write(directory, "status.json", {"status": status, "model": model})
        self.write(directory, "config.json", {"model": {"vital_only": True},
                                              "data": {"horizon": 6}, "training": {"seed": 42}})
        return directory

    def test_neural_reports_and_csv_preserve_missing_threshold_metrics(self):
        directory = self.run_dir()
        self.write(directory, "history.json", [
            {"epoch": 1, "selection_score": .2, "train_loss": .3},
            {"epoch": 2, "selection_score": .1, "train_loss": .2}])
        report = {"metrics": {"n": 10, "positive": 2, "auprc": .2,
                              "threshold": None, "sensitivity": None,
                              "calibration_curve": [{"count": 10}]}}
        for split in ("validation", "test"):
            self.write(directory, f"{split}_report.json", report)
        (directory / "artifact").mkdir()
        self.assertEqual(extract.main(["--runs-dir", str(directory),
                                      "--output-dir", str(self.root / "out")]), 0)
        with (self.root / "out" / "training_results.csv").open(encoding="utf-8-sig", newline="") as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(row["best_epoch"], "1")
        self.assertEqual(row["trained_epochs"], "2")
        self.assertEqual(row["test_positive_rate"], "0.2")
        self.assertEqual(row["test_sensitivity"], "")
        for key in ("test_precision", "test_recall", "test_f1", "test_confusion_matrix",
                    "threshold_strategy", "threshold_selection_precision", "test_fixed_0_5_f1"):
            self.assertEqual(row[key], "")
        result = json.loads((self.root / "out" / "training_results.json").read_text(encoding="utf-8"))
        self.assertEqual(result["runs"][0]["reports"]["test"], report)

    def test_policy_and_evaluation_metrics_remain_distinct_in_mixed_exports(self):
        directory = self.run_dir()
        policy = {"threshold": 0.0, "threshold_strategy": "recall_constraint",
                  "target_recall": .8, "probability_space": "calibrated", "horizon_hours": 6,
                  "classification_rule": "p >= threshold", "unit": "window",
                  "tie_break": "higher_threshold",
                  "selection_metrics": {"n": 8, "precision": .4, "recall": 1., "f1": 4 / 7},
                  "source": {"split": "validation", "role": "threshold_validation",
                             "independent_of_model_and_calibrator_fit": True,
                             "selection_metrics_are_optimization_set_statistics": True,
                             "total_windows": 10, "excluded_windows": 2}}
        self.write(directory, "threshold.json", policy)
        report = {"metrics": {"n": 10, "positive": 2, "threshold": 0.0,
                              "precision": .2, "recall": 1., "f1": 1 / 3,
                              "tp": 2, "tn": 0, "fp": 8, "fn": 0,
                              "confusion_matrix": [[0, 8], [0, 2]],
                              "confusion_matrix_order": "[[TN, FP], [FN, TP]]"},
                  "fixed_0_5": {"threshold": .5, "precision": 0., "recall": 0.,
                                "f1": 0., "confusion_matrix": [[8, 0], [2, 0]]},
                  "calibrated": True, "metric_unit": "window",
                  "continuous_warning_metrics": {"status": "not_computed",
                      "missing_fields": ["event_time", "monitor_end"], "reason": "Missing exposure"}}
        self.write(directory, "test_report.json", report)
        self.write(directory, "validation_report.json", {**report, "validation_role": "early_stopping_selection"})
        self.run_dir("gru")  # Old or unfinished runs share the same CSV schema.
        output = self.root / "out"
        self.assertEqual(extract.main(["--runs-dir", str(self.root / "runs"),
                                      "--output-dir", str(output)]), 0)
        with (output / "training_results.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = {row["model"]: row for row in csv.DictReader(stream)}
        row = rows["tcn"]
        self.assertEqual(row["threshold"], "0.0")
        self.assertEqual(row["threshold_strategy"], "recall_constraint")
        self.assertEqual(row["target_recall"], "0.8")
        self.assertEqual(row["threshold_probability_space"], "calibrated")
        self.assertEqual(row["threshold_horizon_hours"], "6")
        self.assertEqual(row["threshold_source_role"], "threshold_validation")
        self.assertEqual(row["threshold_source_excluded_windows"], "2")
        self.assertEqual(row["threshold_selection_n"], "8")
        self.assertEqual(row["threshold_selection_precision"], "0.4")
        self.assertEqual(row["test_precision"], "0.2")
        self.assertEqual(row["test_recall"], "1.0")
        self.assertAlmostEqual(float(row["test_f1"]), 1 / 3)
        self.assertEqual(json.loads(row["test_confusion_matrix"]), [[0, 8], [0, 2]])
        self.assertEqual(row["test_fixed_0_5_f1"], "0.0")
        self.assertEqual(row["test_fixed_0_5_threshold"], "0.5")
        self.assertEqual(json.loads(row["test_fixed_0_5_confusion_matrix"]), [[8, 0], [2, 0]])
        self.assertEqual(row["validation_validation_role"], "early_stopping_selection")
        self.assertEqual(row["test_continuous_warning_status"], "not_computed")
        self.assertEqual(json.loads(row["test_continuous_warning_missing_fields"]), ["event_time", "monitor_end"])
        self.assertEqual(rows["gru"]["test_f1"], "")
        result = json.loads((output / "training_results.json").read_text(encoding="utf-8"))
        record = next(record for record in result["runs"] if record["summary"]["model"] == "tcn")
        self.assertEqual(record["threshold_policy"], policy)
        self.assertEqual(record["reports"]["test"], report)

    def test_saved_policy_fallbacks_and_precedence(self):
        directory = self.run_dir()
        artifact_policy = {"threshold": .1, "threshold_strategy": "fixed"}
        report_policy = {"threshold": .2, "threshold_strategy": "max_f1"}
        summary_policy = {"threshold": .3, "threshold_strategy": "recall_constraint"}
        root_policy = {"threshold": .4, "threshold_strategy": "fixed"}
        self.write(directory / "artifact", "threshold.json", artifact_policy)
        self.assertEqual(extract.extract_run(directory)["threshold_policy"], artifact_policy)
        self.write(directory, "run_summary.json", {"reports": {"test": {
            "threshold_policy": report_policy, "metrics": {"threshold": .2, "f1": .6}}}})
        record = extract.extract_run(directory)
        self.assertEqual(record["threshold_policy"], report_policy)
        self.assertEqual(record["summary"]["test_f1"], .6)
        self.write(directory, "run_summary.json", {"threshold_policy": summary_policy})
        self.assertEqual(extract.extract_run(directory)["threshold_policy"], summary_policy)
        self.write(directory, "threshold.json", root_policy)
        self.assertEqual(extract.extract_run(directory)["threshold_policy"], root_policy)
        (directory / "threshold.json").write_text("{", encoding="utf-8")
        record = extract.extract_run(directory)
        self.assertEqual(record["threshold_policy"], summary_policy)
        self.assertIn("threshold.json", record["summary"]["extraction_issues"])

    def test_news2_keeps_score_threshold_without_inventing_policy_or_f1(self):
        directory = self.run_dir("news2")
        self.write(directory, "test_report.json", {"metrics": {"threshold": 5, "ppv": .4},
            "continuous_warning_metrics": "not_computed_requires_event_and_exposure_contract"})
        record = extract.extract_run(directory)
        row = record["summary"]
        self.assertEqual(row["test_threshold"], 5)
        self.assertEqual(row["test_ppv"], .4)
        self.assertIsNone(row["test_precision"])
        self.assertIsNone(row["test_f1"])
        self.assertIsNone(row["threshold_strategy"])
        self.assertEqual(record["threshold_policy"], {})
        self.assertEqual(row["test_continuous_warning_status"],
                         "not_computed_requires_event_and_exposure_contract")

    def test_xgboost_iteration_is_zero_based_and_summary_report_fallback(self):
        directory = self.run_dir("xgboost")
        self.write(directory, "history.json", {"validation": {"auprc_ap": [.1, .3, .2]}})
        self.write(directory, "run_summary.json", {"best_iteration": 1, "reports": {
            "test": {"metrics": {"n": 3, "positive": 1, "auprc": .3}}}})
        row = extract.extract_run(directory)["summary"]
        self.assertEqual((row["best_iteration"], row["best_round"], row["trained_rounds"]), (1, 2, 3))
        self.assertEqual(row["test_auprc"], .3)
        self.assertIn("missing validation report", row["extraction_issues"])

    def test_latest_retains_failure_and_separates_horizons(self):
        self.run_dir(stamp="20261101_010000_000000")
        newer = self.run_dir(status="failed")
        self.write(newer, "history.json", {"validation": {}})
        newest = self.run_dir(stamp="20261102_020000_000000")
        self.write(newest, "config.json", {"model": {"vital_only": True},
                                           "data": {"horizon": 12}, "training": {"seed": 42}})
        records = extract.collect_results(self.root / "runs", latest=True)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["summary"]["status"], "failed")

    def test_corrupt_status_does_not_imply_completed(self):
        directory = self.run_dir()
        (directory / "status.json").write_text("{", encoding="utf-8")
        (directory / "artifact").mkdir()
        row = extract.extract_run(directory)["summary"]
        self.assertEqual(row["status"], "unknown")
        self.assertIn("status.json", row["extraction_issues"])


if __name__ == "__main__":
    unittest.main()
