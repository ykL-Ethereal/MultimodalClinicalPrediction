from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

from ..accelerator import autocast_context, resolve_device, resolve_precision
from ..data.collate import move_batch
from ..data.fixed_dataset import FixedPreparedDataset as PreparedDataset
from ..evaluation.calibration import PlattCalibrator, sigmoid
from ..evaluation.thresholds import apply_policy
from ..preprocessing.text import make_backend
from ..preprocessing.text_cache import TextCache
from ..training.checkpoint import load_artifact
from ..training.trainer import make_loader


class Predictor:
    def __init__(self, artifact, *, device="auto", cache_dir=None, precision=None):
        self.path = Path(artifact)
        self.model, self.config, self.contract, self.preprocessor, self.manifest = load_artifact(artifact, device)
        self.is_xgb = self.config.model.vital_encoder == "xgboost"
        self.device = torch.device("cpu") if self.is_xgb else resolve_device(device)
        self.precision = "float32" if self.is_xgb else resolve_precision(self.device, precision or self.config.training.precision)
        self.backend = self.cache = None
        if not self.config.model.vital_only:
            self.config.text.device = str(self.device)
            self.backend = make_backend(self.config.text, snapshot=self.path / "text_backbone",
                                        expected_identity=self.manifest["text_identity"])
            self.cache = TextCache(cache_dir or self.path.parent / "inference_cache", self.backend)
        calibration = self.manifest["calibrator"]
        self.calibrator = PlattCalibrator(**calibration) if calibration else None
        self.policy = self.manifest["threshold_policy"]
        self.probability_space = "calibrated" if self.calibrator else "raw"

    @torch.inference_mode()
    def predict(self, bundle, *, rows=None, batch_size=16, diagnostics=False):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        from ..data.fixed_adapter import FixedCohortBundle
        if isinstance(bundle, (str, Path)):
            bundle = FixedCohortBundle(bundle, vital_only=self.config.model.vital_only, require_targets=False)
        elif self.config.model.vital_only and not bundle.vital_only:
            bundle = FixedCohortBundle(bundle.root, vital_only=True, require_targets=False)
        self.contract.assert_matches(bundle.contract)
        rows = np.arange(len(bundle.metadata)) if rows is None else np.asarray(rows)
        if rows.ndim != 1 or (len(rows) and rows.dtype.kind not in "iu") or (rows < 0).any() or (rows >= len(bundle.metadata)).any():
            raise ValueError("Invalid inference row indices")
        rows = rows.astype(np.int64)
        if len(rows) == 0:
            return []
        results = []
        if not self.is_xgb:
            self.model.eval()
        for start in range(0, len(rows), batch_size):
            selected_rows = rows[start:start + batch_size]
            if self.is_xgb:
                from ..models.vital_xgboost import predict_margins
                inputs = self.preprocessor.transform(bundle, selected_rows)
                logits = predict_margins(self.model, inputs["vitals"])
                probabilities = self.calibrator.predict(logits) if self.calibrator else sigmoid(logits)
                for i, row in enumerate(selected_rows):
                    results.append(self._vital_result(bundle.metadata.iloc[row], probabilities[i], inputs["available"][i, 0]))
                continue
            dataset = PreparedDataset(bundle, selected_rows, self.preprocessor, self.cache,
                                      self.config.data.horizon, with_targets=False,
                                      defer_numeric=self.device.type == "cuda")
            batch = move_batch(next(iter(make_loader(dataset, len(dataset), device=self.device))), self.device)
            with autocast_context(self.device, self.precision):
                output = self.model(batch["inputs"])
            logits = output["logits"].double().cpu().numpy()
            probabilities = self.calibrator.predict(logits) if self.calibrator else sigmoid(logits)
            valid_rows = output["valid"].cpu().tolist()
            used_rows = output["used_modalities"].cpu().tolist()
            gate_rows = output["gate_weights"].float().cpu().tolist() if diagnostics else None
            if self.config.model.vital_only:
                for i, meta in enumerate(batch["metadata"]):
                    results.append(self._vital_result(meta, probabilities[i], valid_rows[i]))
                continue
            fixed_lab_masks = bundle.get_numeric(selected_rows)["lab_mask"][:,:,0]
            for i, meta in enumerate(batch["metadata"]):
                valid = valid_rows[i]
                used = used_rows[i]
                active = np.asarray(self.preprocessor.active_lab_features)
                lab_mask = fixed_lab_masks[i]
                disabled = int(np.sum(lab_mask & ~active))
                item = {"sample_id": str(meta["sample_id"]), "horizon_hours": self.config.data.horizon,
                        "risk": float(probabilities[i]) if valid else None,
                        "status": "ok" if valid else "invalid_input_no_dynamic_modality",
                        "used_modalities": [name for name, present in zip(("vitals", "labs", "notes"), used) if present],
                        "calibrated": self.calibrator is not None, "text_truncated": False,
                        "disabled_observed_lab_count": disabled, "unknown_lab_policy": "reject",
                        "model_version": self.manifest["model_version"],
                        "preprocessing_version": self.preprocessor.identifier,
                        "purpose": "research_prediction"}
                item.update(self._decision(probabilities[i], valid))
                if diagnostics and self.config.model.fusion == "gated":
                    item["gate_weights"] = gate_rows[i]
                results.append(item)
        return results

    def _vital_result(self, meta, probability, valid):
        item = {"sample_id": str(meta["sample_id"]), "horizon_hours": self.config.data.horizon,
                "risk": float(probability) if valid else None,
                "status": "ok" if valid else "invalid_input_no_vitals",
                "used_modalities": ["vitals"] if valid else [], "calibrated": self.calibrator is not None,
                "model_version": self.manifest["model_version"],
                "preprocessing_version": self.preprocessor.identifier, "purpose": "research_prediction"}
        item.update(self._decision(probability, valid))
        return item

    def _decision(self, probability, valid):
        prediction = int(apply_policy([probability], self.policy, probability_space=self.probability_space,
                                      horizon_hours=self.config.data.horizon)[0]) if valid else None
        return {"threshold": self.policy["threshold"], "probability_space": self.probability_space,
                "prediction": prediction, "alarm": bool(prediction) if valid else None}

