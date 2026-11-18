"""Stream a fixed cohort into a new package; never modify the source package."""
from copy import deepcopy
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from ..io import file_sha256, fingerprint, read_json, write_json
from .fixed_adapter import FixedCohortBundle


def spaced_rows(frame, minutes=30):
    """First timestamp per encounter, then earliest timestamp >= last + interval.

    Only encounter identity and timestamp determine inclusion. Gaps are not filled,
    labels and event times cannot affect the selected phase or retention.
    """
    if type(minutes) is not int or minutes <= 0:
        raise ValueError("minutes must be a positive integer")
    times = pd.to_datetime(frame.prediction_time, errors="raise").astype("int64").to_numpy() // 10**9
    if pd.isna(frame.encounter_id).any():
        raise ValueError("Missing encounter identity")
    selected = []
    for positions in frame.groupby("encounter_id", sort=False).indices.values():
        ordered = positions[np.argsort(times[positions], kind="stable")]
        if len(np.unique(times[ordered])) != len(ordered):
            raise ValueError("Duplicate prediction timestamp within encounter")
        last = None
        for row in ordered:
            if last is None or times[row] - last >= minutes * 60:
                selected.append(int(row))
                last = times[row]
    return np.asarray(sorted(selected), dtype=np.int64)


def cohort_counts(frame):
    result = {}
    for split, group in frame.groupby("split", sort=True):
        result[split] = {"windows": len(group), "patients": int(group.patient_group_id.nunique()),
                         "admissions": int(group.encounter_id.nunique())}
        for h in (1, 2, 3, 6):
            positive = group[f"label_{h}h"].eq(1)
            result[split][f"positive_windows_{h}h"] = int(positive.sum())
            result[split][f"positive_patients_{h}h"] = int(group.loc[positive, "patient_group_id"].nunique())
    return result


def resample_cohort(source, destination, minutes=30):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or destination.is_relative_to(source) or destination.exists():
        raise ValueError("Choose a new output directory outside the original package")
    bundle = FixedCohortBundle(source)
    frame = pd.read_csv(source / "sample_index.csv.gz", dtype={"patient_group_id": str, "encounter_id": str})
    rows = spaced_rows(frame, minutes)
    chosen = frame.iloc[rows].copy()
    chosen["source_sample_id"] = chosen.sample_id.to_numpy()
    chosen["sample_id"] = np.arange(len(chosen))
    assert set(chosen.patient_group_id) == set(frame.patient_group_id)
    destination.mkdir(parents=True)
    (destination / "tensors").mkdir()
    (destination / "splits").mkdir()
    write_json(destination / "resampling_status.json", {"status": "running"})
    schema = deepcopy(bundle.schema)
    schema["shards"] = []
    hashes = {}
    offset = 0
    for sid, spec in enumerate(bundle.schema["shards"]):
        local = rows[(rows >= spec["start"]) & (rows < spec["start"] + spec["rows"])] - spec["start"]
        if not len(local):
            continue
        source_file = source / spec["file"]
        hashes[spec["file"]] = file_sha256(source_file)
        with np.load(source_file, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        bundle._validate_shard(arrays, spec)
        if any(a.ndim == 0 or len(a) != spec["rows"] for a in arrays.values()):
            raise ValueError("All tensor fields must be row-aligned")
        arrays = {k: a[local] for k, a in arrays.items()}
        name = f"tensors/shard_{len(schema['shards']):06d}.npz"
        np.savez_compressed(destination / name, **arrays)
        # Verify every written field, including NaN placeholders, against the source.
        with np.load(destination / name, allow_pickle=False) as z:
            for key, value in arrays.items():
                if z[key].dtype != value.dtype or not np.array_equal(z[key], value, equal_nan=True):
                    raise ValueError(f"Resampling changed tensor values: {name}/{key}")
        schema["shards"].append({"file": name, "start": offset, "rows": len(local)})
        offset += len(local)
        if sid % 200 == 0:
            print(f"resample shards={sid + 1}/{len(bundle.schema['shards'])} retained={offset}", flush=True)
    chosen.to_csv(destination / "sample_index.csv.gz", index=False)
    for split in ("train", "validation", "test"):
        np.save(destination / "splits" / f"{split}_sample_ids.npy",
                chosen.loc[chosen.split == split, "sample_id"].to_numpy(np.int64), allow_pickle=False)
    for name in ("patient_splits.csv", "lab_feature_catalog.csv", "note_text.jsonl", "note_index.npz"):
        shutil.copyfile(source / name, destination / name)
    counts = cohort_counts(chosen)
    schema.update(samples=len(chosen), version=f"{schema['version']}-stride{minutes}m",
                  sampling_interval_minutes=minutes,
                  sampling_rule="first per encounter then earliest >= previous retained time + interval; label blind")
    if "split" in schema:
        schema["split"]["counts"] = [{"split": key, **value} for key, value in counts.items()]
    provenance = {"source_root": str(source), "source_schema_sha256": file_sha256(source / "dataset_schema.json"),
                  "source_index_sha256": file_sha256(source / "sample_index.csv.gz"),
                  "patient_splits_sha256": file_sha256(source / "patient_splits.csv"),
                  "source_shard_hashes": hashes, "selected_source_rows_sha256": fingerprint(rows.tolist()),
                  "interval_minutes": minutes, "history_minutes": 120, "selection_uses_labels": False,
                  "before": cohort_counts(frame), "after": counts}
    write_json(destination / "resampling_provenance.json", provenance)
    schema["resampling_provenance_sha256"] = file_sha256(destination / "resampling_provenance.json")
    write_json(destination / "dataset_schema.json", schema)
    verified = FixedCohortBundle(destination)
    for sid in range(len(verified.starts)):
        verified.shard(sid)
    write_json(destination / "resampling_status.json", {"status": "completed", "all_tensor_fields_verified": True,
                                                       "counts": counts})
    return destination
