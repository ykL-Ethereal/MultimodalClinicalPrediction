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


class CUDAMoments(Moments):
    """Bounded FP64 reductions on CUDA; serialize the same host statistics."""
    def __init__(self, size, device):
        import torch
        self.torch, self.device = torch, device
        self.count = torch.zeros(size, dtype=torch.int64, device=device)
        self.mean = torch.zeros(size, dtype=torch.float64, device=device)
        self.m2 = torch.zeros_like(self.mean)

    def add(self, values, mask):
        torch = self.torch
        # Shards can grow independently of GPU memory. Only transfer small blocks.
        for start in range(0, len(values), 1024):
            shape = (-1, len(self.count))
            v = torch.as_tensor(np.ascontiguousarray(values[start:start + 1024]), device=self.device).reshape(shape)
            m = torch.as_tensor(np.ascontiguousarray(mask[start:start + 1024]), device=self.device).reshape(shape)
            count = m.sum(dim=0)
            x = v.masked_fill(~m, 0).double()
            mean = x.sum(dim=0) / count.clamp(min=1)
            m2 = ((x - mean) ** 2).masked_fill(~m, 0).sum(dim=0)
            total = self.count + count
            delta = mean - self.mean
            self.m2 += m2 + delta ** 2 * self.count * count / total.clamp(min=1)
            self.mean += delta * count / total.clamp(min=1)
            self.count = total

    def to_host(self):
        self.count = self.count.cpu().numpy()
        self.mean = self.mean.cpu().numpy()
        self.m2 = self.m2.cpu().numpy()


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
    def fit(cls, bundle, training_rows, *, device="cpu"):
        rows = np.asarray(training_rows, dtype=np.int64)
        if len(rows) == 0 or not np.isin(rows, bundle.rows("train")).all():
            raise ValueError("Preprocessing may only fit nonempty training rows")
        if len(np.unique(rows)) != len(rows):
            raise ValueError("Duplicate training rows")
        if str(device) != "cpu":
            from ..accelerator import resolve_device
            device = resolve_device(device)
        moments = (lambda size: CUDAMoments(size, device)) if str(device).startswith("cuda") else Moments
        vital, lab, age = moments(6), moments(len(bundle.catalog)), moments(1)
        for _, a in bundle.iter_numeric(rows):
            observed = a["vital_observed_mask"].copy()
            observed[:,:,4:] &= a["vital_features"][:,:,[10,13]] > 0
            vital.add(a["vital_features"][:,:,VALUE_CHANNELS], observed)
            lab.add(a["lab_features"][:,:,0], a["lab_mask"][:,:,0])
            age.add(a["static_features"][:,:1], a["static_mask"][:,:1])
        for accumulator in (vital, lab, age):
            if isinstance(accumulator, CUDAMoments):
                accumulator.to_host()
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

    def transform_tensors(self, a):
        """Normalize a collated raw batch on its target device, before autocast.

        Disk reads and tokenization stay on the host. Numeric values, masks,
        log ages and modality quality are computed where the model runs.
        """
        import math
        import torch
        device = a["vital_features"].device
        cache = getattr(self, "_device_parameters", {})
        if device not in cache:
            cache[device] = {name: torch.tensor(getattr(self, name), dtype=torch.float64, device=device)
                             for name in ("vital_mean", "vital_std", "lab_mean", "lab_std")}
            cache[device].update(
                active=torch.tensor(self.active_lab_features, dtype=torch.bool, device=device),
                supported=torch.tensor(self.fit_provenance["vital_observed_counts"], device=device) >= 2)
            self._device_parameters = cache
        p = cache[device]

        def age(values, mask, scale, cap=None):
            x = values.masked_fill(~mask, 0).double()
            if cap is not None:
                x = x.clamp(max=cap)
            return (x.log1p() / math.log1p(scale)).float()

        v = a["vital_features"].clone()
        observed = a["vital_observed_mask"].clone()
        present = torch.cat([(v[:, :, :4] != 0) | observed[:, :, :4], v[:, :, [10, 13]] > 0], dim=2)
        standardized = (v[:, :, VALUE_CHANNELS].double() - p["vital_mean"]) / p["vital_std"]
        observed &= present & p["supported"]
        v[:, :, VALUE_CHANNELS] = standardized.masked_fill(~(present & p["supported"]), 0).float()
        vitals = torch.cat([v, observed.float()], dim=2)
        lm = a["lab_mask"] & p["active"][None, :, None]
        values = torch.empty_like(a["lab_features"])
        values[:, :, 0] = ((a["lab_features"][:, :, 0].double() - p["lab_mean"]) / p["lab_std"]).float()
        values[:, :, 1] = (a["lab_features"][:, :, 1].double() / p["lab_std"]).float()
        la = age(a["lab_age_hours"], lm[:, :, 0], 72, 72)
        interval = age(a["lab_interval_hours"], lm[:, :, 1], 72)
        labs = torch.cat([values.masked_fill(~lm, 0), lm.float(), la[..., None], interval[..., None]], dim=2)
        static = a["static_features"].masked_fill(~a["static_mask"], 0).float()
        static[:, 0] = ((static[:, 0] - self.age_mean[0]) / self.age_std[0]).masked_fill(~a["static_mask"][:, 0], 0)
        static = torch.cat([static, a["static_mask"].float()], dim=1)
        nm = a["note_mask"]
        available = torch.stack([present.any(dim=(1, 2)), lm[:, :, 0].any(dim=1), nm.any(dim=1)], dim=1)
        vq = torch.stack([present.any(dim=2).float().mean(dim=1), observed.float().mean(dim=(1, 2)),
                          observed[:, -15:].any(dim=1).float().mean(dim=1)], dim=1)
        nactive = p["active"].sum().clamp(min=1)
        lq = torch.stack([lm[:, :, 0].sum(dim=1) / nactive, lm[:, :, 1].sum(dim=1) / nactive,
                          la.sum(dim=1) / lm[:, :, 0].sum(dim=1).clamp(min=1)], dim=1)
        note_age = age(a["note_age"], nm, 24)
        nearest = note_age.masked_fill(~nm, torch.inf).amin(dim=1)
        nq = torch.stack([nm.sum(dim=1).float().log1p(), nearest.masked_fill(~available[:, 2], 0),
                          torch.zeros(len(nm), device=device)], dim=1)
        result = {"vitals": vitals.float(), "time_mask": torch.ones(v.shape[:2], dtype=torch.bool, device=device),
                "labs": labs.flatten(1).float(), "static": static, "available": available,
                "quality": torch.stack([vq, lq, nq], dim=1).masked_fill(~available[..., None], 0).float(),
                "note_age": note_age, "note_mask": nm}
        finite = torch.stack([torch.isfinite(x).all() for x in result.values()])
        if not finite.all():
            raise ValueError("Non-finite transformed fixed-cohort inputs")
        return {**result, "note_chunks": a["note_chunks"], "chunk_mask": a["chunk_mask"]}
