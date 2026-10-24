from pathlib import Path
import numpy as np

from ..data.fixed_adapter import FixedCohortBundle
from ..io import write_json
from ..preprocessing.fixed_numeric import FixedNumericPreprocessor
from ..preprocessing.text import make_backend
from ..preprocessing.text_cache import TextCache


def inspect_bundle(bundle, horizon):
    if horizon not in bundle.horizons:
        raise ValueError(f"Package horizons are {bundle.horizons}, got {horizon}")
    report = {"samples": len(bundle.metadata), "vitals": len(bundle.contract.vital_names),
              "lab_features": len(bundle.contract.lab_names), "horizon_hours": horizon, "splits": {}}
    for split in ("train", "validation", "test"):
        rows = bundle.rows(split)
        y, mask = bundle.labels(rows, horizon)
        report["splits"][split] = {"samples": len(rows), "patients": int(bundle.metadata.iloc[rows]["patient_id"].nunique()),
            "labelled": int(mask.sum()), "positive": int(np.sum(y[mask] == 1)), "negative": int(np.sum(y[mask] == 0))}
    return report


def prepare(config, *, encode_text=True):
    config.validate()
    root = Path(config.data.root)
    if not (root / "dataset_schema.json").is_file():
        raise FileNotFoundError(f"Fixed data package not found: {root / 'dataset_schema.json'}")
    bundle = FixedCohortBundle(root)
    report = inspect_bundle(bundle, config.data.horizon)
    length = bundle.schema["history_minutes"]
    if length > config.model.max_time_steps:
        raise ValueError("Input time length exceeds model max_time_steps")
    preprocessor = FixedNumericPreprocessor.fit(bundle, bundle.rows("train"))
    output = Path(config.training.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if not encode_text:
        report.update(bundle.validate_all())
    report["active_labs"] = sum(preprocessor.active_lab_features)
    write_json(output / "data_report.json", report)
    write_json(output / "preprocessing_preview.json", preprocessor.to_dict())
    if not encode_text:
        return bundle, preprocessor, None, None
    backend = make_backend(config.text)
    cache = TextCache(config.text.cache_dir, backend)
    return bundle, preprocessor, backend, cache

