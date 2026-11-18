"""Unnormalized, split-specific disk memmaps shared across patient folds."""
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset

from ..io import file_sha256, fingerprint, read_json, write_json


def cached_split(bundle, split, directory):
    rows = bundle.rows(split)
    specs = [bundle.schema["shards"][sid] for sid in np.unique(bundle.shard_ids(rows))]
    identity = {"root": str(bundle.root), "split": split,
                "schema": file_sha256(bundle.root / "dataset_schema.json"),
                "index": file_sha256(bundle.root / "sample_index.csv.gz"),
                "files": [(s["file"], (bundle.root / s["file"]).stat().st_size,
                           (bundle.root / s["file"]).stat().st_mtime_ns) for s in specs]}
    path = Path(directory) / fingerprint(identity)
    ready = path / "ready.json"
    if ready.exists():
        if fingerprint(read_json(ready)["identity"]) != fingerprint(identity):
            raise ValueError("Numeric cache identity mismatch")
        expected = {"values.npy": ((len(rows), 120, 14), np.dtype("float32")),
                    "observed.npy": ((len(rows), 120, 6), np.dtype("bool")),
                    "available.npy": ((len(rows),), np.dtype("bool"))}
        for name, (shape, dtype) in expected.items():
            array = np.load(path / name, mmap_mode="r", allow_pickle=False)
            if array.shape != shape or array.dtype != dtype:
                raise ValueError("Corrupt numeric cache")
            array._mmap.close()
        return path, rows
    path.mkdir(parents=True, exist_ok=True)
    print(f"building unnormalized {split} memmap: {len(rows)} rows", flush=True)
    values = np.lib.format.open_memmap(path / "values.npy", mode="w+", dtype="float32", shape=(len(rows), 120, 14))
    observed = np.lib.format.open_memmap(path / "observed.npy", mode="w+", dtype="bool", shape=(len(rows), 120, 6))
    available = np.zeros(len(rows), bool)
    for selected, arrays in bundle.iter_numeric(rows):
        dest = np.searchsorted(rows, selected)
        v, o = arrays["vital_features"], arrays["vital_observed_mask"]
        values[dest], observed[dest] = v, o
        available[dest] = ((v[:, :, :4] != 0).any(axis=(1, 2)) | o.any(axis=(1, 2))
                           | (v[:, :, [10, 13]] > 0).any(axis=(1, 2)))
    values.flush()
    observed.flush()
    del values, observed
    np.save(path / "available.npy", available, allow_pickle=False)
    write_json(ready, {"identity": identity, "rows": len(rows), "normalized": False})
    return path, rows


class CachedVitalDataset(Dataset):
    """Workers reopen read-only memmaps, never copy the tensor cohort into RAM."""
    def __init__(self, bundle, rows, preprocessor, horizon, directory, *, defer_numeric=False):
        self.rows = np.asarray(rows, dtype=np.int64)
        splits = bundle.metadata.iloc[self.rows].split.unique()
        if len(splits) != 1:
            raise ValueError("A cached dataset must belong to one original split")
        self.path, cached_rows = cached_split(bundle, splits[0], directory)
        self.local = np.searchsorted(cached_rows, self.rows)
        if not np.array_equal(cached_rows[self.local], self.rows):
            raise ValueError("Cache rows do not match requested rows")
        self.y, self.label_mask = bundle.labels(self.rows, horizon)
        self.available = np.zeros((len(rows), 3), bool)
        self.available[:, 0] = np.load(self.path / "available.npy", allow_pickle=False)[self.local]
        self.metadata = bundle.metadata.iloc[self.rows].reset_index(drop=True).copy()
        self.preprocessor, self.defer_numeric = preprocessor, defer_numeric
        self._values = self._observed = None
        # transform_arrays only needs the schema for its contract assertion.
        from types import SimpleNamespace
        self.contract_bundle = SimpleNamespace(schema=bundle.schema)

    def __len__(self):
        return len(self.rows)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_values"] = state["_observed"] = None
        return state

    def close(self):
        """Release Windows mmap handles when the caller finishes with this dataset."""
        for name in ("_values", "_observed"):
            array = getattr(self, name, None)
            if array is not None:
                array._mmap.close()
                setattr(self, name, None)

    def __getitem__(self, index):
        return self.__getitems__([index])[0]

    def __getitems__(self, indices):
        if self._values is None:
            self._values = np.load(self.path / "values.npy", mmap_mode="r", allow_pickle=False)
            self._observed = np.load(self.path / "observed.npy", mmap_mode="r", allow_pickle=False)
        positions = self.local[indices]
        arrays = {"vital_features": self._values[positions], "vital_observed_mask": self._observed[positions]}
        inputs = arrays if self.defer_numeric else self.preprocessor.transform_arrays(self.contract_bundle, arrays)
        records = self.metadata.iloc[indices][["sample_id", "patient_id", "episode_id"]].to_dict("records")
        samples = [{"inputs": {k: v[i] for k, v in inputs.items()}, "target": float(self.y[index]),
                    "label_mask": bool(self.label_mask[index]), "metadata": records[i]}
                   for i, index in enumerate(indices)]
        if self.defer_numeric:
            for sample in samples:
                sample["numeric_preprocessor"] = self.preprocessor
        return samples
