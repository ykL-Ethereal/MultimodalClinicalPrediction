"""Speed settings preserve sample order, preprocessing and saved decisions."""
from pathlib import Path

import numpy as np
import pytest
import torch

from clinical_multimodal.accelerator import configure_tf32
from clinical_multimodal.config import Config
from clinical_multimodal.data.fixed_adapter import FixedCohortBundle
from clinical_multimodal.data.fixed_dataset import FixedPreparedDataset
from clinical_multimodal.inference.predictor import Predictor
from clinical_multimodal.io import read_json
from clinical_multimodal.pipelines.train import train_pipeline
from clinical_multimodal.preprocessing.vital_numeric import VitalNumericPreprocessor
from clinical_multimodal.preprocessing.text import MacBertBackend
from tests.fixed_fixture import make_fixed
from tests.helpers import TestWorkspace
from tests.test_vital_only import configuration


def test_metadata_fast_path_keeps_tensor_order_and_duplicate_rows():
    with TestWorkspace() as directory:
        bundle = FixedCohortBundle(make_fixed(Path(directory) / "data"), vital_only=True)
        processor = VitalNumericPreprocessor.fit(bundle, bundle.rows("train"))
        rows = np.array([5, 0, 5, 2])
        regular = FixedPreparedDataset(bundle, rows, processor, None, 6).__getitems__([0, 1, 2, 3])
        lean = FixedPreparedDataset(bundle, rows, processor, None, 6,
                                   include_metadata=False).__getitems__([0, 1, 2, 3])
        assert [s["metadata"]["sample_id"] for s in regular] == rows.tolist()
        for a, b in zip(regular, lean):
            assert b["metadata"] == {}
            assert a["target"] == b["target"] and a["label_mask"] == b["label_mask"]
            for key in a["inputs"]:
                np.testing.assert_array_equal(a["inputs"][key], b["inputs"][key])


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_vital_parameter_cache_matches_reference_and_is_not_serialized(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    with TestWorkspace() as directory:
        bundle = FixedCohortBundle(make_fixed(Path(directory) / "data"), vital_only=True)
        processor = VitalNumericPreprocessor.fit(bundle, bundle.rows("train"))
        serialized = processor.to_dict()
        raw = bundle.get_numeric([5, 0, 5, 2])
        expected = processor.transform_arrays(bundle, raw)
        inputs = {k: torch.as_tensor(raw[k], device=device) for k in ("vital_features", "vital_observed_mask")}
        processor.transform_tensors(inputs)
        cached = processor._device_parameters[torch.device(device)]
        actual = processor.transform_tensors(inputs)
        assert processor._device_parameters[torch.device(device)] is cached
        for key in expected:
            torch.testing.assert_close(actual[key].cpu(), torch.from_numpy(expected[key]), atol=2e-6, rtol=2e-6)
        assert processor.to_dict() == serialized


def test_tf32_can_be_reset_between_runs():
    try:
        for enabled in (True, False):
            configure_tf32(enabled)
            if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
                assert torch.backends.cuda.matmul.fp32_precision == ("tf32" if enabled else "ieee")
                assert torch.backends.cudnn.conv.fp32_precision == ("tf32" if enabled else "ieee")
            else:
                assert torch.backends.cuda.matmul.allow_tf32 == enabled
    finally:
        configure_tf32(False)
    with pytest.raises(ValueError, match="eval_batch_size"):
        Config.from_dict({"training": {"eval_batch_size": -1}})


def test_text_cache_identity_separates_tf32_without_loading_backbone():
    backend = MacBertBackend.__new__(MacBertBackend)
    backend.torch, backend.identity = torch, {"weights": "fixture"}
    backend.device, backend.precision = torch.device("cuda:0"), "float32"
    try:
        configure_tf32(False)
        original = backend.cache_identity
        configure_tf32(True)
        assert backend.cache_identity != original
        assert backend.cache_identity["matmul_tf32"] is True
        configure_tf32(False)
        assert backend.cache_identity == original
    finally:
        configure_tf32(False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fast_cuda_training_with_distinct_eval_batch_and_saved_threshold():
    with TestWorkspace() as directory:
        workspace = Path(directory)
        root = make_fixed(workspace / "data", validation_patients=9)
        cfg = configuration(workspace)
        cfg.training.device, cfg.training.precision = "cuda:0", "auto"
        cfg.training.deterministic, cfg.training.allow_tf32 = False, True
        cfg.training.batch_size, cfg.training.eval_batch_size = 4, 3
        cfg.training.threshold_strategy, cfg.training.threshold = "max_f1", None
        try:
            artifact = train_pipeline(cfg)
            summary = read_json(artifact.parent / "run_summary.json")
            assert summary["allow_tf32"] and summary["eval_batch_size"] == 3
            assert summary["threshold_policy"]["threshold_strategy"] == "max_f1"
            assert summary["threshold_policy"]["target_recall"] is None
            history = read_json(artifact.parent / "history.json")
            assert history[0]["train_samples_per_second"] > 0
            assert history[0]["validation_seconds"] > 0
            predictor = Predictor(artifact, device="cuda:0")
            result = predictor.predict(root, rows=[5, 0, 5], batch_size=3)
            assert [r["sample_id"] for r in result] == ["5", "0", "5"]
            assert all(np.isfinite(r["risk"]) for r in result)
            assert result[0]["prediction"] == result[2]["prediction"]
        finally:
            configure_tf32(False)
