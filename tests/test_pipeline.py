from copy import deepcopy
from pathlib import Path
import os
import subprocess
import sys
import unittest
import numpy as np
import torch
import yaml

from clinical_multimodal.config import Config, load_config
from clinical_multimodal.evaluation.calibration import PlattCalibrator, sigmoid
from clinical_multimodal.evaluation.metrics import binary_metrics
from clinical_multimodal.inference.predictor import Predictor
from clinical_multimodal.io import read_json
from clinical_multimodal.pipelines.evaluate import evaluate_pipeline
from clinical_multimodal.pipelines.train import train_pipeline, validation_partition
from clinical_multimodal.training.checkpoint import load_artifact
from tests.helpers import fixture, small_config, TestWorkspace


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = TestWorkspace()
        cls.root, cls.bundle, cls.processor = fixture(cls.temp.name)
        cls.config = small_config(cls.temp.name)
        cls.artifacts = {}
        for name in ('tcn', 'lstm', 'gru', 'grud'):
            cfg = deepcopy(cls.config)
            cfg.model.vital_encoder = name
            cls.artifacts[name] = train_pipeline(cfg)
        cls.artifact = cls.artifacts['tcn']
        cls.predictor = Predictor(cls.artifact)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_all_models_reload_equal_selected_weights_and_predictions(self):
        for name, artifact in self.artifacts.items():
            with self.subTest(model=name):
                predictor = Predictor(artifact)
                self.assertEqual(predictor.config.model.vital_encoder, name)
                selected = torch.load(artifact.parent / 'best.pt', map_location='cpu', weights_only=True)
                for key, value in predictor.model.state_dict().items():
                    torch.testing.assert_close(value, selected[key], atol=0, rtol=0)
                a = predictor.predict(self.bundle, rows=[5, 0], batch_size=2)
                b = Predictor(artifact).predict(self.bundle, rows=[5, 0], batch_size=1)
                np.testing.assert_allclose([x['risk'] for x in a], [x['risk'] for x in b], rtol=1e-5, atol=1e-6)
                self.assertEqual([x['sample_id'] for x in a], ['5', '0'])
                self.assertTrue(all(0 <= x['risk'] <= 1 for x in a))

    def test_logs_reports_and_config_are_separate(self):
        self.assertEqual(len(set(self.artifacts.values())), 4)
        for name, artifact in self.artifacts.items():
            run = artifact.parent
            self.assertEqual(run.parent.name, name)
            self.assertIn('epoch=1', (run / 'training.log').read_text(encoding='utf-8'))
            self.assertEqual(read_json(run / 'status.json')['status'], 'completed')
            self.assertEqual(read_json(run / 'config.json')['model']['vital_encoder'], name)
            self.assertEqual(read_json(run / 'run_summary.json')['vital_encoder'], name)
            self.assertEqual(read_json(run / 'test_report.json')['metrics']['status'], 'ok')

    def test_independent_process_artifact_only_prediction(self):
        output = Path(self.temp.name) / 'independent.json'
        code = ("from clinical_multimodal.data.fixed_adapter import FixedCohortBundle; "
                "from clinical_multimodal.inference.predictor import Predictor; "
                "from clinical_multimodal.io import write_json; import sys; "
                "write_json(sys.argv[3], Predictor(sys.argv[1]).predict(FixedCohortBundle(sys.argv[2]), rows=[0,5]))")
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / 'src'))
        run = subprocess.run([sys.executable, '-B', '-c', code, str(self.artifacts['grud']), str(self.root), str(output)],
                             env=env, capture_output=True, text=True, timeout=90)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(len(read_json(output)), 2)

    def test_main_runs_with_configuration_only(self):
        cfg = deepcopy(self.config)
        cfg.training.models = ['gru']
        config_path = Path(self.temp.name) / 'train.yaml'
        config_path.write_text(yaml.safe_dump(cfg.to_dict(), allow_unicode=True), encoding='utf-8')
        project = Path(__file__).resolve().parents[1]
        run = subprocess.run([sys.executable, '-B', str(project / 'main.py'), '--config', str(config_path)],
                             capture_output=True, text=True, timeout=90, cwd=self.temp.name)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn('model=gru', run.stdout)
        self.assertEqual(len(list((Path(cfg.training.output_dir) / 'gru').glob('*/artifact'))), 2)

    def test_failed_run_keeps_error_log(self):
        cfg = deepcopy(self.config)
        cfg.data.root = str(Path(self.temp.name) / 'missing')
        with self.assertRaises(FileNotFoundError):
            train_pipeline(cfg)
        failures = [p for p in Path(cfg.training.output_dir).glob('tcn/*/status.json')
                    if read_json(p)['status'] == 'failed']
        self.assertEqual(len(failures), 1)
        self.assertIn('FileNotFoundError', failures[0].with_name('training.log').read_text(encoding='utf-8'))

    def test_metadata_and_targets_cannot_change_prediction(self):
        b = deepcopy(self.bundle)
        b.targets = {}
        b.metadata['patient_id'] = 'changed'
        b.metadata['future_event'] = 1000000
        a = self.predictor.predict(self.bundle, rows=[0, 1])
        modified = self.predictor.predict(b, rows=[0, 1])
        self.assertEqual([x['risk'] for x in a], [x['risk'] for x in modified])

    def test_all_missing_returns_null_risk(self):
        b = deepcopy(self.bundle)
        a = b.shard(0)
        a['vital_features'][:] = 0
        a['vital_observed_mask'][:] = False
        a['lab_mask'][:] = False
        a['note_end'][:] = a['note_start']
        result = self.predictor.predict(b, rows=[0])[0]
        self.assertIsNone(result['risk'])
        self.assertEqual(result['used_modalities'], [])
        self.assertEqual(result['status'], 'invalid_input_no_dynamic_modality')

    def test_evaluation_report(self):
        report = evaluate_pipeline(self.artifact, self.root, Path(self.temp.name) / 'evaluation.json')
        self.assertEqual(report['metrics']['status'], 'ok')
        self.assertIsNotNone(report['metrics']['auroc'])

    def test_manifest_and_weights_integrity(self):
        import shutil
        destination = Path(self.temp.name) / 'tampered'
        shutil.copytree(self.artifact, destination)
        with (destination / 'weights.pt').open('ab') as f:
            f.write(b'corruption')
        with self.assertRaisesRegex(ValueError, 'integrity'):
            load_artifact(destination)

    def test_calibration_requires_distinct_patients(self):
        cfg = deepcopy(self.config)
        cfg.training.calibration = True
        with self.assertRaisesRegex(ValueError, 'two validation patients'):
            validation_partition(self.bundle, cfg)
        # Test the partition contract with two distinct validation patients.
        bundle = deepcopy(self.bundle)
        bundle.metadata.loc[3, 'patient_id'] = 'OTHER'
        selection, calibration = validation_partition(bundle, cfg)
        self.assertTrue(set(bundle.metadata.iloc[selection].patient_id).isdisjoint(bundle.metadata.iloc[calibration].patient_id))
        self.assertEqual(sorted(np.concatenate([selection, calibration])), bundle.rows('validation').tolist())

    def test_single_class_metrics_are_not_zero(self):
        metrics = binary_metrics(np.zeros(3), np.array([.1,.2,.3]), threshold=.25)
        self.assertIsNone(metrics['auprc'])
        self.assertIsNone(metrics['auroc'])
        self.assertEqual(metrics['status'], 'single_class')
        self.assertIsNone(binary_metrics([], [])['brier'])

    def test_platt_and_stable_sigmoid(self):
        x = np.array([-3,-2,-1,1,2,3.])
        y = np.array([0,0,1,0,1,1])
        self.assertTrue(np.isfinite(PlattCalibrator.fit(x,y).predict(x)).all())
        np.testing.assert_array_equal(sigmoid(np.array([-1e9,1e9])), [0,1])

    def test_rerun_creates_new_directory(self):
        second = train_pipeline(self.config)
        self.assertNotEqual(second, self.artifact)
        self.assertTrue(self.artifact.is_dir())

    def test_default_config_resolves_packaged_data(self):
        root = Path(__file__).resolve().parents[1]
        cfg = load_config(root / 'configs/train.yaml')
        self.assertEqual(Path(cfg.data.root), root / 'data/Mutimodel_data/固定训练数据')
        self.assertTrue((Path(cfg.data.root) / 'dataset_schema.json').is_file())
        self.assertEqual(cfg.training.models, ['tcn', 'lstm', 'gru'])
        for raw in ({'data': {'horizon':24}}, {'text': {'backend':'demo'}}, {'data': {'mode':'raw_records'}},
                    {'training': {'models':['gru','gru']}}, {'model': {'vital_encoder':'unknown'}}):
            with self.assertRaises(ValueError):
                Config.from_dict(raw)
        Config.from_dict({'model': {'vital_encoder':'gru', 'tcn_blocks':1}})
