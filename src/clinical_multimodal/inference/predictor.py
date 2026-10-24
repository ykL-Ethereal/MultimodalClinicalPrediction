from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

from ..data.collate import move_batch
from ..data.fixed_dataset import FixedPreparedDataset as PreparedDataset
from ..evaluation.calibration import PlattCalibrator, sigmoid
from ..preprocessing.text import make_backend
from ..preprocessing.text_cache import TextCache
from ..training.checkpoint import load_artifact
from ..training.trainer import make_loader


class Predictor:
    def __init__(self, artifact, *, device="cpu", cache_dir=None):
        self.path = Path(artifact)
        self.device = torch.device(device)
        self.model, self.config, self.contract, self.preprocessor, self.manifest = load_artifact(artifact, device)
        self.config.text.device = device
        self.backend = make_backend(self.config.text, snapshot=self.path / "text_backbone",
                                    expected_identity=self.manifest["text_identity"])
        self.cache = TextCache(cache_dir or self.path.parent / "inference_cache", self.backend)
        calibration = self.manifest["calibrator"]
        self.calibrator = PlattCalibrator(**calibration) if calibration else None

    @torch.no_grad()
    def predict(self, bundle, *, rows=None, batch_size=16, diagnostics=False):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.contract.assert_matches(bundle.contract)
        rows = np.arange(len(bundle.metadata)) if rows is None else np.asarray(rows, dtype=int)
        if rows.ndim != 1 or (rows < 0).any() or (rows >= len(bundle.metadata)).any():
            raise ValueError("Invalid inference row indices")
        if len(rows) == 0:
            return []
        results = []
        self.model.eval()
        for start in range(0, len(rows), batch_size):
            selected_rows = rows[start:start + batch_size]
            dataset = PreparedDataset(bundle, selected_rows, self.preprocessor, self.cache,
                                      self.config.data.horizon, with_targets=False)
            batch = move_batch(next(iter(make_loader(dataset, len(dataset)))), self.device)
            output = self.model(batch["inputs"])
            logits = output["logits"].cpu().numpy()
            probabilities = self.calibrator.predict(logits) if self.calibrator else sigmoid(logits)
            fixed_lab_masks = bundle.get_numeric(selected_rows)["lab_mask"][:,:,0]
            for i, meta in enumerate(batch["metadata"]):
                valid = bool(output["valid"][i])
                used = output["used_modalities"][i].cpu().tolist()
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
                threshold = self.manifest["threshold"]
                if threshold is not None:
                    item.update(threshold=threshold, alarm=bool(probabilities[i] >= threshold) if valid else None)
                if diagnostics and self.config.model.fusion == "gated":
                    item["gate_weights"] = output["gate_weights"][i].cpu().tolist()
                results.append(item)
        return results

