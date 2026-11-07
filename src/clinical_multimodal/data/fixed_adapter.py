"""Read the final PAT/ENC cohort without materializing all window tensors."""
from collections import OrderedDict
from pathlib import Path
import json

import numpy as np
import pandas as pd

from ..io import fingerprint, read_json
from .schema import Contract
from .validation import require


VITAL_ORDER = ["hr", "pr", "rr", "spo2", "hr_mask", "pr_mask", "rr_mask", "spo2_mask",
               "sbp_last", "sbp_mins_since_last_norm", "sbp_last_mask",
               "dbp_last", "dbp_mins_since_last_norm", "dbp_last_mask"]
VALUE_CHANNELS = [0, 1, 2, 3, 8, 11]
SPLIT_CODES = {"train": 0, "validation": 1, "test": 2}


def vital_contract(schema):
    """The vital-only contract does not depend on excluded modalities."""
    return Contract(schema["vital_feature_order"].copy(), [], [], [],
                    fingerprint({"normalization": schema["normalization"],
                                 "history": schema["history_minutes"]}),
                    schema_version="fixed-cohort-vital-v1")


class FixedCohortBundle:
    is_sharded = True

    def __init__(self, root, *, require_targets=True, vital_only=False):
        self.root = Path(root).resolve()
        self.vital_only = vital_only
        self.schema = s = read_json(self.root / "dataset_schema.json")
        require(s["horizons_hours"] == [1, 2, 3, 6], "Unexpected fixed-cohort horizons")
        require(s["vital_feature_order"] == VITAL_ORDER, "Unexpected vital feature order")
        require(s["history_minutes"] == 120, "Unexpected history length")
        require(s["normalization"] == "none; fit on training patients after splitting",
                "Fixed cohort must contain unstandardized values")
        self.horizons, self.split_codes = s["horizons_hours"], SPLIT_CODES.copy()
        if vital_only:
            self.catalog = pd.DataFrame()
            self.contract = vital_contract(s)
        else:
            require(s["lab_channels"] == ["latest_value", "latest_minus_previous"], "Unexpected lab channels")
            require(s["static_feature_order"] == ["age_years", "sex_female", "sex_male"], "Unexpected static order")
            self.catalog = pd.read_csv(self.root / "lab_feature_catalog.csv", dtype={"code": str, "unit": str}, keep_default_na=False)
            require(self.catalog.feature_index.tolist() == list(range(s["lab_feature_count"])), "Invalid lab feature indices")
            require(not self.catalog.duplicated(["code", "unit"]).any(), "Duplicate lab code/unit")
            self.contract = Contract(VITAL_ORDER.copy(),
                [json.dumps([r.code, r.unit], ensure_ascii=False) for r in self.catalog.itertuples()],
                s["static_feature_order"], s["lab_channels"],
                fingerprint({"normalization": s["normalization"], "history": 120, "lab_lookback": s["lab_lookback_hours"]}),
                schema_version="fixed-cohort-v1")
        self.normalization = {"source": "unstandardized", "fit_split": "train",
                              "parameters_location": "preprocessing.json"}
        self.metadata = m = pd.read_csv(self.root / "sample_index.csv.gz",
            dtype={"patient_group_id": str, "encounter_id": str})
        require(len(m) == s["samples"] and np.array_equal(m.sample_id, np.arange(len(m))),
                "sample_id must equal the global tensor row")
        require(m[["patient_group_id", "encounter_id", "split", "prediction_time"]].notna().all().all(),
                "Missing cohort identity/time/split")
        require(m.split.isin(SPLIT_CODES).all(), "Unknown split")
        require(m.groupby("patient_group_id").split.nunique().max() == 1, "Patient leakage across splits")
        require(m.groupby("encounter_id").patient_group_id.nunique().max() == 1, "Encounter maps to multiple patients")
        require(m.patient_group_id.nunique() == s["patients"] and m.encounter_id.nunique() == s["admissions"],
                "Schema patient/admission totals disagree with index")
        self.prediction_seconds = pd.to_datetime(m.prediction_time, errors="raise").astype("int64").to_numpy() // 10**9
        m["patient_id"] = m.patient_group_id
        m["episode_id"] = m.encounter_id
        m["tensor_row"] = m.sample_id
        patients = pd.read_csv(self.root / "patient_splits.csv", dtype=str)
        require(patients.patient_group_id.is_unique and patients.split.isin(SPLIT_CODES).all(), "Invalid patient_splits")
        expected = m.patient_group_id.map(patients.set_index("patient_group_id").split)
        require(expected.equals(m.split), "Patient split table disagrees with sample index")
        require(set(patients.patient_group_id) == set(m.patient_group_id), "Patient split table has unmatched patients")
        self.targets = {"split_code": m.split.map(SPLIT_CODES).to_numpy(np.int8)}
        columns = [f"label_{h}h" for h in self.horizons]
        if require_targets or all(c in m for c in columns):
            require(all(c in m for c in columns), "Missing supervised labels")
            labels = m[columns].to_numpy()
            require(np.isin(labels, [0, 1]).all(), "Invalid fixed-cohort labels")
            self.targets.update(y=labels.astype(np.int8), label_mask=np.ones(labels.shape, bool))
        for split in SPLIT_CODES:
            ids = np.load(self.root / "splits" / f"{split}_sample_ids.npy", allow_pickle=False)
            require(np.array_equal(ids, self.rows(split)), f"Stored {split} IDs disagree with index")
        self.starts = np.asarray([x["start"] for x in s["shards"]], dtype=np.int64)
        sizes = np.asarray([x["rows"] for x in s["shards"]], dtype=np.int64)
        require(len(sizes) > 0 and (sizes > 0).all() and np.array_equal(self.starts, np.r_[0, sizes.cumsum()[:-1]])
                and sizes.sum() == len(m), "Shards must cover every row exactly once")
        for shard in s["shards"]:
            path = (self.root / shard["file"]).resolve()
            require(path.is_relative_to(self.root) and path.is_file(), "Missing/unsafe shard path")
        if vital_only:
            self.offsets = self.note_seconds = np.empty(0, dtype=np.int64)
        else:
            with np.load(self.root / "note_index.npz", allow_pickle=False) as z:
                self.offsets, self.note_seconds = z["offsets"], z["seconds"]
        require(self.offsets.shape == self.note_seconds.shape == (0 if vital_only else s["notes"],), "Invalid note index shape")
        require(self.offsets.dtype.kind in "iu" and self.note_seconds.dtype.kind in "iu", "Invalid note index dtype")
        require((self.offsets >= 0).all() and (np.diff(self.offsets) > 0).all(), "Invalid note offsets")
        if len(self.offsets):
            require(self.offsets[-1] < (self.root / "note_text.jsonl").stat().st_size, "Note offset outside file")
        self._shards, self._notes = OrderedDict(), OrderedDict()
        self._available = np.zeros((len(m), 3), bool)
        self._validated = np.zeros(len(sizes), bool)

    def rows(self, split):
        return np.flatnonzero(self.targets["split_code"] == self.split_codes[split])

    def labels(self, rows, horizon):
        if horizon not in self.horizons:
            raise ValueError(f"This package supports horizons {self.horizons}, got {horizon}")
        head = self.horizons.index(horizon)
        return self.targets["y"][rows, head], self.targets["label_mask"][rows, head]

    def shard_ids(self, rows):
        return np.searchsorted(self.starts, rows, side="right") - 1

    def shard(self, sid):
        sid = int(sid)
        if sid in self._shards:
            self._shards.move_to_end(sid)
            return self._shards[sid]
        spec = self.schema["shards"][sid]
        with np.load(self.root / spec["file"], allow_pickle=False) as z:
            keys = ["vital_features", "vital_observed_mask", "prediction_second"] if self.vital_only else z.files
            if self.vital_only and "y" in self.targets:
                keys.append("y")
            a = {k: z[k] for k in keys}
        self._validate_shard(a, spec)
        self._validated[sid] = True
        rows = slice(spec["start"], spec["start"] + spec["rows"])
        self._available[rows, 0] = ((a["vital_features"][:,:,:4] != 0).any(axis=(1,2)) |
            a["vital_observed_mask"].any(axis=(1, 2)) |
            (a["vital_features"][:, :, [10, 13]] > 0).any(axis=(1, 2)))
        if not self.vital_only:
            self._available[rows, 1] = a["lab_mask"][:, :, 0].any(axis=1)
            self._available[rows, 2] = a["note_end"] > a["note_start"]
        self._shards[sid] = a
        while len(self._shards) > 2:
            self._shards.popitem(last=False)
        return a

    def _validate_shard(self, a, spec):
        n, f = spec["rows"], len(self.catalog)
        shapes = {"vital_features": (n,120,14), "vital_observed_mask": (n,120,6),
            "lab_features": (n,f,2), "lab_mask": (n,f,2), "lab_age_hours": (n,f),
            "lab_interval_hours": (n,f), "static_features": (n,3), "static_mask": (n,3),
            "note_start": (n,), "note_end": (n,), "prediction_second": (n,)}
        if self.vital_only:
            shapes = {k: shapes[k] for k in ("vital_features", "vital_observed_mask", "prediction_second")}
        for key, shape in shapes.items():
            require(key in a and a[key].shape == shape, f"Invalid {key} shape in {spec['file']}")
        for key in (("vital_observed_mask",) if self.vital_only else ("vital_observed_mask", "lab_mask", "static_mask")):
            require(a[key].dtype == np.bool_, f"Invalid {key} dtype")
        v = a["vital_features"]
        require(np.isfinite(v).all(), "Non-finite vital features")
        require(np.isin(v[:, :, [4,5,6,7,10,13]], [0,1]).all(), "Invalid vital indicator")
        require(np.array_equal(v[:,:,4:8] == 0, a["vital_observed_mask"][:,:,:4]), "Vital missing/observed masks disagree")
        require(((v[:,:,[9,12]] >= 0) & (v[:,:,[9,12]] <= 1)).all(), "Invalid blood pressure age")
        first = spec["start"]
        require(a["prediction_second"].dtype.kind in "iu", "Invalid prediction_second dtype")
        require(np.array_equal(a["prediction_second"], self.prediction_seconds[first:first+n]), "Prediction time misalignment")
        if "y" in self.targets:
            require("y" in a and np.array_equal(a["y"], self.targets["y"][first:first+n]), "Shard/index label misalignment")
        if self.vital_only:
            return
        for key, mask in (("lab_features", a["lab_mask"]), ("static_features", a["static_mask"])):
            require(np.isfinite(a[key][mask]).all(), f"Non-finite observed {key}")
        require((~a["lab_mask"][:,:,1] | a["lab_mask"][:,:,0]).all(), "Lab delta requires a latest value")
        for key, mask in (("lab_age_hours", a["lab_mask"][:,:,0]), ("lab_interval_hours", a["lab_mask"][:,:,1])):
            require((np.isfinite(a[key][mask]) & (a[key][mask] >= 0)).all(), f"Invalid {key}")
        require((a["lab_age_hours"][a["lab_mask"][:,:,0]] <= 72.001).all(), "Lab exceeds lookback")
        for key in ("note_start", "note_end", "prediction_second"):
            require(a[key].dtype.kind in "iu", f"Invalid {key} dtype")
        start, end = a["note_start"], a["note_end"]
        require(((0 <= start) & (start <= end) & (end <= len(self.offsets))).all(), "Invalid note range")
        for lo, hi, t in zip(start, end, a["prediction_second"]):
            require((self.note_seconds[lo:hi] <= t).all(), "Future document in input")

    def iter_numeric(self, rows):
        rows = np.asarray(rows, dtype=np.int64)
        ids = self.shard_ids(rows)
        # Rows are normally sorted; grouping once avoids repeatedly scanning the full cohort.
        order = np.argsort(ids, kind="stable")
        grouped = rows[order]
        boundaries = np.r_[0, np.flatnonzero(np.diff(ids[order])) + 1, len(rows)]
        for lo, hi in zip(boundaries[:-1], boundaries[1:]):
            if lo == hi:
                continue
            chosen = grouped[lo:hi]
            sid = int(self.shard_ids(chosen[:1])[0])
            a = self.shard(sid)
            local = chosen - self.starts[sid]
            yield chosen, {k: v[local] for k, v in a.items()}

    def get_numeric(self, rows):
        rows = np.asarray(rows, dtype=np.int64)
        require(rows.ndim == 1 and len(rows) > 0 and ((rows >= 0) & (rows < len(self.metadata))).all(), "Invalid row indices")
        ids, result = self.shard_ids(rows), {}
        for sid in np.unique(ids):
            dest = np.flatnonzero(ids == sid)
            local = rows[dest] - self.starts[sid]
            for key, values in self.shard(sid).items():
                if key not in result:
                    result[key] = np.empty((len(rows),) + values.shape[1:], values.dtype)
                result[key][dest] = values[local]
        return result

    def note(self, index):
        index = int(index)
        if index in self._notes:
            self._notes.move_to_end(index)
            return self._notes[index]
        with (self.root / "note_text.jsonl").open("rb") as f:
            f.seek(int(self.offsets[index]))
            text = json.loads(f.readline())
        require(isinstance(text, str) and bool(text.strip()), "Empty/non-string valid document")
        self._notes[index] = text
        while len(self._notes) > 512:
            self._notes.popitem(last=False)
        return text

    def materialize(self, rows):
        a = self.get_numeric(rows)
        d = max(1, int((a["note_end"] - a["note_start"]).max()))
        mask, ages, texts = np.zeros((len(rows), d), bool), np.zeros((len(rows), d), np.float32), []
        for i, (lo, hi, t) in enumerate(zip(a["note_start"], a["note_end"], a["prediction_second"])):
            texts.append([self.note(j) for j in range(lo, hi)] + [""] * (d - (hi - lo)))
            mask[i, :hi-lo] = True
            ages[i, :hi-lo] = (t - self.note_seconds[lo:hi]) / 3600
        a.update(note_doc_mask=mask, note_age_hours=ages)
        return a, texts

    def availability(self, rows):
        for sid in np.unique(self.shard_ids(rows)):
            if not self._validated[sid]:
                self.shard(sid)
        return self._available[rows]

    def validate_all(self):
        for sid in range(len(self.starts)):
            self.shard(sid)
        for index in range(len(self.offsets)):
            self.note(index)
        return {"validated_shards": len(self.starts), "validated_documents": len(self.offsets)}
