"""Streaming training-only standardization of the unnormalized fixed cohort."""
from dataclasses import asdict, dataclass
import numpy as np

from ..data.fixed_adapter import VALUE_CHANNELS
from ..data.schema import Contract
from ..io import fingerprint
from .numeric_utils import log_age, masked


class Moments:
    def __init__(self, size):
        self.count = np.zeros(size, np.int64)
        self.mean = np.zeros(size, np.float64)
        self.m2 = np.zeros(size, np.float64)

    def add(self, values, mask):
        values, mask = values.reshape(-1, len(self.count)), mask.reshape(-1, len(self.count))
        count = mask.sum(axis=0)
        x = np.where(mask, values, 0).astype(np.float64)
        mean = x.sum(axis=0) / count.clip(min=1)
        m2 = np.where(mask, (x - mean) ** 2, 0).sum(axis=0)
        total = self.count + count
        delta = mean - self.mean
        self.m2 += m2 + delta ** 2 * self.count * count / total.clip(min=1)
        self.mean += delta * count / total.clip(min=1)
        self.count = total

    def parameters(self):
        std = np.sqrt(self.m2 / self.count.clip(min=1))
        std[(std < 1e-6) | (self.count < 2)] = 1
        return self.mean.tolist(), std.tolist()


@dataclass
class FixedNumericPreprocessor:
    contract: dict
    active_lab_features: list[bool]
    vital_mean: list[float]
    vital_std: list[float]
    lab_mean: list[float]
    lab_std: list[float]
    age_mean: list[float]
    age_std: list[float]
    fit_provenance: dict
    version: str = "fixed-numeric-v1"
    output_vital_channels: int = 20

    @classmethod
    def fit(cls, bundle, training_rows):
        rows = np.asarray(training_rows, dtype=np.int64)
        if len(rows) == 0 or not np.isin(rows, bundle.rows("train")).all():
            raise ValueError("Preprocessing may only fit nonempty training rows")
        if len(np.unique(rows)) != len(rows):
            raise ValueError("Duplicate training rows")
        vital, lab, age = Moments(6), Moments(len(bundle.catalog)), Moments(1)
        for _, a in bundle.iter_numeric(rows):
            observed = a["vital_observed_mask"].copy()
            observed[:,:,4:] &= a["vital_features"][:,:,[10,13]] > 0
            vital.add(a["vital_features"][:,:,VALUE_CHANNELS], observed)
            lab.add(a["lab_features"][:,:,0], a["lab_mask"][:,:,0])
            age.add(a["static_features"][:,:1], a["static_mask"][:,:1])
        vm, vs = vital.parameters()
        lm, ls = lab.parameters()
        am, ast = age.parameters()
        provenance = {"fit_split": "train", "samples": len(rows),
            "patient_ids": sorted(bundle.metadata.iloc[rows].patient_id.unique().tolist()),
            "sample_ids_sha256": fingerprint(bundle.metadata.iloc[rows].sample_id.tolist()),
            "vital_observed_counts": vital.count.tolist(), "lab_latest_counts": lab.count.tolist(),
            "age_count": int(age.count[0]), "weighting": "window positions; overlapping windows contribute repeatedly",
            "variance": "population", "unsupported_labs": "disable when fewer than two training latest values"}
        return cls(bundle.contract.to_dict(), (lab.count >= 2).tolist(), vm, vs, lm, ls, am, ast, provenance)

    def to_dict(self):
        return asdict(self)

    @property
    def identifier(self):
        return fingerprint(self.to_dict())

    def transform(self, bundle, rows):
        a, _ = bundle.materialize(rows)
        return self.transform_arrays(bundle, a)

    def transform_arrays(self, bundle, a):
        Contract(**self.contract).assert_matches(bundle.contract)
        v = a["vital_features"].copy()
        observed = a["vital_observed_mask"].copy()
        # High-frequency values are causal forward fills; a leading zero is a placeholder.
        present = np.concatenate([(v[:,:,:4] != 0) | observed[:,:,:4], v[:,:,[10,13]] > 0], axis=2)
        standardized = (v[:,:,VALUE_CHANNELS] - np.asarray(self.vital_mean)) / np.asarray(self.vital_std)
        supported = np.asarray(self.fit_provenance["vital_observed_counts"]) >= 2
        observed &= present & supported
        v[:,:,VALUE_CHANNELS] = np.where(present & supported, standardized, 0)
        vitals = np.concatenate([v, observed.astype(np.float32)], axis=2)
        active = np.asarray(self.active_lab_features, bool)
        lm = a["lab_mask"] & active[None,:,None]
        values = np.empty_like(a["lab_features"])
        values[:,:,0] = (a["lab_features"][:,:,0] - np.asarray(self.lab_mean)) / np.asarray(self.lab_std)
        values[:,:,1] = a["lab_features"][:,:,1] / np.asarray(self.lab_std)
        la = log_age(a["lab_age_hours"], lm[:,:,0], 72, 72)
        interval = log_age(a["lab_interval_hours"], lm[:,:,1], 72)
        labs = np.concatenate([masked(values, lm), lm.astype(np.float32), la[...,None], interval[...,None]], axis=2)
        static = masked(a["static_features"], a["static_mask"])
        static[:,0] = np.where(a["static_mask"][:,0], (static[:,0] - self.age_mean[0]) / self.age_std[0], 0)
        static = np.concatenate([static, a["static_mask"]], axis=1).astype(np.float32)
        nm = a["note_doc_mask"]
        available = np.stack([present.any(axis=(1,2)), lm[:,:,0].any(axis=1), nm.any(axis=1)], axis=1)
        vq = np.stack([present.any(axis=2).mean(axis=1), observed.mean(axis=(1,2)),
                       observed[:,-15:].any(axis=1).mean(axis=1)], axis=1)
        nactive = max(int(active.sum()), 1)
        lq = np.stack([lm[:,:,0].sum(axis=1)/nactive, lm[:,:,1].sum(axis=1)/nactive,
                       la.sum(axis=1)/lm[:,:,0].sum(axis=1).clip(min=1)], axis=1)
        note_age = log_age(a["note_age_hours"], nm, 24)
        nearest = np.where(nm, note_age, np.inf).min(axis=1)
        nq = np.stack([np.log1p(nm.sum(axis=1)), np.where(available[:,2], nearest, 0), np.zeros(len(nm))], axis=1)
        result = {"vitals": vitals.astype(np.float32), "time_mask": np.ones(v.shape[:2],bool),
            "labs": labs.reshape(len(v),-1).astype(np.float32), "static": static,
            "available": available, "quality": np.where(available[...,None], np.stack([vq,lq,nq],axis=1),0).astype(np.float32),
            "note_age": note_age, "note_mask": nm}
        if any(not np.isfinite(x).all() for x in result.values()):
            raise ValueError("Non-finite transformed fixed-cohort inputs")
        return result
