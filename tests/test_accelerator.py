"""Actual CUDA training, mixed precision, caching and device transfer regressions."""
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from clinical_multimodal.accelerator import resolve_device, resolve_precision
from clinical_multimodal.config import Config
from clinical_multimodal.data.collate import move_batch
from clinical_multimodal.data.fixed_dataset import FixedPreparedDataset
from clinical_multimodal.inference.predictor import Predictor
from clinical_multimodal.io import read_json
from clinical_multimodal.models.predictor import MultimodalRiskModel
from clinical_multimodal.pipelines.train import train_pipeline
from clinical_multimodal.preprocessing.fixed_numeric import FixedNumericPreprocessor
from clinical_multimodal.preprocessing.text import MacBertBackend
from clinical_multimodal.preprocessing.text_cache import TextCache
from clinical_multimodal.training.trainer import Trainer, make_loader, seed_everything
from tests.helpers import TestWorkspace, fixture, small_config


def test_cpu_fallback_and_explicit_cuda_failure():
    with patch('torch.cuda.is_available', return_value=False):
        assert resolve_device('auto') == torch.device('cpu')
        assert resolve_precision(resolve_device('auto'), 'auto') == 'float32'
        with pytest.raises(RuntimeError, match='CUDA'):
            resolve_device('cuda:0')
    for field, value in [('device', 'cuda:-1'), ('precision', 'fp7')]:
        with pytest.raises(ValueError):
            Config.from_dict({'training': {field: value}})


cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires CUDA')


@cuda_only
@pytest.mark.parametrize('encoder', ['tcn', 'lstm', 'gru', 'grud'])
def test_cuda_amp_pipeline_and_portable_artifact(encoder):
    with TestWorkspace() as directory:
        _, bundle, cpu_processor = fixture(directory)
        cfg = small_config(directory)
        cfg.model.vital_encoder = encoder
        cfg.training.device = cfg.text.device = 'auto'
        cfg.training.precision = cfg.text.precision = 'auto'
        cfg.training.deterministic = False
        artifact = train_pipeline(cfg)
        summary = read_json(artifact.parent / 'run_summary.json')
        assert summary['device'].startswith('cuda:')
        assert summary['precision'] in ('bfloat16', 'float16')
        assert summary['performance']['training_evaluation_cuda_peak_allocated_bytes'] > 0
        predictor = Predictor(artifact)
        assert next(predictor.model.parameters()).is_cuda
        assert next(predictor.backend.model.parameters()).is_cuda
        for name in ('vital_mean', 'vital_std', 'lab_mean', 'lab_std', 'age_mean', 'age_std'):
            np.testing.assert_allclose(getattr(cpu_processor, name), getattr(predictor.preprocessor, name), atol=1e-12)
        # Compare FP32 CUDA/CPU independently of AMP rounding and batch shape.
        gpu = Predictor(artifact, precision='float32')
        cpu = Predictor(artifact, device='cpu', precision='float32')
        a = gpu.predict(bundle, rows=[0, 5], diagnostics=True)
        b = cpu.predict(bundle, rows=[0, 5], diagnostics=True)
        np.testing.assert_allclose([p['risk'] for p in a], [p['risk'] for p in b], atol=2e-4, rtol=2e-4)
        assert all(np.isfinite(p['risk']) for p in predictor.predict(bundle, rows=[0, 5]))
        assert a[0]['used_modalities'] == b[0]['used_modalities']
        state = torch.load(artifact.parent / 'last.pt', map_location='cpu', weights_only=True)
        assert 'scaler' in state


@cuda_only
def test_cuda_fp16_scaler_pinned_batches_and_text_cache_precision():
    with TestWorkspace() as directory:
        _, bundle, processor = fixture(directory)
        cfg = small_config(directory)
        cfg.training.device, cfg.training.precision = 'cuda:0', 'float16'
        cfg.text.device, cfg.text.precision = 'cuda:0', 'float32'
        seed_everything(7, 1)
        backend = MacBertBackend(cfg.text)
        cache = TextCache(Path(directory) / 'cache', backend)
        text = '病情稳定。' * 20
        key = cache.key(text)
        fp32 = cache.get(text)
        backend.precision = 'float16'
        assert cache.key(text) != key
        fp16 = cache.get(text)
        assert fp32.shape == fp16.shape and np.isfinite(fp16).all()
        np.testing.assert_array_equal(fp16, cache.get(text))
        dataset = FixedPreparedDataset(bundle, bundle.rows('train'), processor, cache, 6, defer_numeric=True)
        loader = make_loader(dataset, 2, device='cuda:0')
        raw = next(iter(loader))
        assert raw['target'].is_pinned()
        assert all(v.is_pinned() for v in raw['inputs'].values())
        batch = move_batch(raw, 'cuda:0')
        assert all(v.is_cuda for v in batch['inputs'].values())
        model = MultimodalRiskModel(len(bundle.contract.vital_names), len(bundle.contract.lab_names),
                                    backend.dimension, cfg.model, vital_channels=20)
        trainer = Trainer(model, cfg.training, Path(directory))
        before = next(trainer.model.parameters()).detach().clone()
        result = trainer.train_batch(raw)
        assert result is not None and np.isfinite(result).all()
        assert trainer.scaler.is_enabled()
        assert trainer.optimizer.defaults['fused']
        assert not torch.equal(before, next(trainer.model.parameters()))


@cuda_only
def test_cuda_numeric_transform_matches_masked_cpu_reference():
    with TestWorkspace() as directory:
        _, bundle, processor = fixture(directory)
        cfg = small_config(directory)
        backend = MacBertBackend(cfg.text)
        cache = TextCache(Path(directory) / 'cache', backend)
        # Include empty notes, active/inactive labs, and missing physical features.
        rows = np.array([5, 0, 2, 1, 5])
        cpu_dataset = FixedPreparedDataset(bundle, rows, processor, cache, 6)
        gpu_dataset = FixedPreparedDataset(bundle, rows, processor, cache, 6, defer_numeric=True)
        cpu = next(iter(make_loader(cpu_dataset, len(rows))))['inputs']
        gpu = move_batch(next(iter(make_loader(gpu_dataset, len(rows), device='cuda:0'))), 'cuda:0')['inputs']
        assert cpu.keys() == gpu.keys()
        for name in cpu:
            assert gpu[name].is_cuda
            torch.testing.assert_close(gpu[name].cpu(), cpu[name], atol=2e-6, rtol=2e-6)
        # Verify fillers and NaNs under missing masks are suppressed on CUDA.
        raw = next(iter(make_loader(gpu_dataset, len(rows))))
        raw['inputs']['lab_mask'].fill_(False)
        raw['inputs']['lab_features'].fill_(float('nan'))
        result = move_batch(raw, 'cuda:0')['inputs']
        assert torch.isfinite(result['labs']).all()
        assert not result['available'][:, 1].any()
