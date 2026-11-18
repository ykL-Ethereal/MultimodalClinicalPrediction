"""Training-only vital statistics and the same 20-channel representation as multimodal runs."""
from dataclasses import asdict, dataclass
import numpy as np

from ..data.fixed_adapter import VALUE_CHANNELS, vital_contract
from ..data.schema import Contract
from ..io import fingerprint
from .fixed_numeric import Moments, CUDAMoments


@dataclass
class VitalNumericPreprocessor:
    contract: dict
    vital_mean: list[float]
    vital_std: list[float]
    fit_provenance: dict
    version: str = "vital-numeric-v1"
    output_vital_channels: int = 20
    vital_only = True

    @classmethod
    def fit(cls, bundle, training_rows, *, device="cpu"):
        rows = np.asarray(training_rows, dtype=np.int64)
        if not len(rows) or not np.isin(rows, bundle.rows("train")).all():
            raise ValueError("Preprocessing may only fit nonempty training rows")
        if len(np.unique(rows)) != len(rows):
            raise ValueError("Duplicate training rows")
        from ..accelerator import resolve_device
        device = resolve_device(device)
        accumulator = CUDAMoments(6, device) if device.type == "cuda" else Moments(6)
        for _, a in bundle.iter_numeric(rows):
            observed = a["vital_observed_mask"].copy()
            observed[:, :, 4:] &= a["vital_features"][:, :, [10, 13]] > 0
            accumulator.add(a["vital_features"][:, :, VALUE_CHANNELS], observed)
        if isinstance(accumulator, CUDAMoments):
            accumulator.to_host()
        mean, std = accumulator.parameters()
        provenance = {"fit_split": "train", "samples": len(rows),
                      "patient_ids": sorted(bundle.metadata.iloc[rows].patient_id.unique().tolist()),
                      "sample_ids_sha256": fingerprint(bundle.metadata.iloc[rows].sample_id.tolist()),
                      "vital_observed_counts": accumulator.count.tolist(),
                      "weighting": "window positions; overlapping windows contribute repeatedly",
                      "variance": "population"}
        return cls(vital_contract(bundle.schema).to_dict(), mean, std, provenance)

    def to_dict(self):
        return asdict(self)

    @property
    def identifier(self):
        return fingerprint(self.to_dict())

    def transform(self, bundle, rows):
        return self.transform_arrays(bundle, bundle.get_numeric(rows))

    def transform_arrays(self, bundle, a):
        Contract(**self.contract).assert_matches(vital_contract(bundle.schema))
        v = a["vital_features"].copy()
        observed = a["vital_observed_mask"].copy()
        present = np.concatenate([(v[:, :, :4] != 0) | observed[:, :, :4], v[:, :, [10, 13]] > 0], axis=2)
        supported = np.asarray(self.fit_provenance["vital_observed_counts"]) >= 2
        observed &= present & supported
        standardized = (v[:, :, VALUE_CHANNELS] - np.asarray(self.vital_mean)) / np.asarray(self.vital_std)
        v[:, :, VALUE_CHANNELS] = np.where(present & supported, standardized, 0)
        result = {"vitals": np.concatenate([v, observed.astype(np.float32)], axis=2).astype(np.float32),
                  "time_mask": np.ones(v.shape[:2], bool),
                  "available": np.stack([present.any(axis=(1, 2)), np.zeros(len(v), bool), np.zeros(len(v), bool)], axis=1)}
        if not np.isfinite(result["vitals"]).all():
            raise ValueError("Non-finite transformed vital inputs")
        return result

    def transform_tensors(self, a):
        import torch
        v = a["vital_features"].clone()
        observed = a["vital_observed_mask"].clone()
        present = torch.cat([(v[:, :, :4] != 0) | observed[:, :, :4], v[:, :, [10, 13]] > 0], dim=2)
        mean = torch.as_tensor(self.vital_mean, dtype=torch.float64, device=v.device)
        std = torch.as_tensor(self.vital_std, dtype=torch.float64, device=v.device)
        supported = torch.as_tensor(self.fit_provenance["vital_observed_counts"], device=v.device) >= 2
        observed &= present & supported
        standardized = (v[:, :, VALUE_CHANNELS].double() - mean) / std
        v[:, :, VALUE_CHANNELS] = standardized.masked_fill(~(present & supported), 0).float()
        available = torch.zeros((len(v), 3), dtype=torch.bool, device=v.device)
        available[:, 0] = present.any(dim=(1, 2))
        vitals = torch.cat([v, observed.float()], dim=2).float()
        if not torch.isfinite(vitals).all():
            raise ValueError("Non-finite transformed vital inputs")
        return {"vitals": vitals,
                "time_mask": torch.ones(v.shape[:2], dtype=torch.bool, device=v.device), "available": available}
