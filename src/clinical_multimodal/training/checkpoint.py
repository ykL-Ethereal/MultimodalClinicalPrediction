"""Self-contained, integrity-checked inference artifact; weights-only deserialization."""
from __future__ import annotations

from pathlib import Path
import platform
import importlib.metadata
import torch

from .. import __version__
from ..accelerator import resolve_device
from ..config import Config
from ..data.schema import Contract
from ..io import file_sha256, fingerprint, read_json, write_json
from ..models.predictor import make_risk_model
from ..preprocessing.fixed_numeric import FixedNumericPreprocessor
from ..preprocessing.vital_numeric import VitalNumericPreprocessor
from ..evaluation.thresholds import validate_policy


def vital_input_layout():
    from ..data.fixed_adapter import VITAL_ORDER
    channels = VITAL_ORDER + [f"{name}_observed" for name in ("hr", "pr", "rr", "spo2", "sbp", "dbp")]
    return {"shape": [120, 20], "flatten_order": "C: time first, channel second",
            "channel_names": channels,
            "feature_names": [f"minute_{t:03d}_{name}" for t in range(120) for name in channels]}


def source_fingerprint():
    source = Path(__file__).resolve().parents[1]
    return fingerprint({str(p.relative_to(source)): file_sha256(p) for p in sorted(source.rglob("*.py"))})


def save_artifact(path, model, config, bundle, preprocessor, backend, *, calibrator=None, training_record=None,
                  threshold_policy=None):
    probability_space = "calibrated" if calibrator else "raw"
    validate_policy(threshold_policy, probability_space=probability_space, horizon_hours=config.data.horizon)
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Artifact already exists: {path}")
    path.mkdir(parents=True)
    is_xgb = config.model.vital_encoder == "xgboost"
    if is_xgb:
        model.save_model(path / "model.ubj")
    else:
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, path / "weights.pt")
    write_json(path / "config.json", config.to_dict())
    write_json(path / "contract.json", bundle.contract.to_dict())
    write_json(path / "normalization.json", bundle.normalization)
    write_json(path / "schema.json", bundle.schema)
    write_json(path / "preprocessing.json", preprocessor.to_dict())
    write_json(path / "threshold.json", threshold_policy)
    if backend is not None:
        backend.export(path / "text_backbone")
    if config.model.vital_only:
        write_json(path / "input_layout.json", vital_input_layout())
    packages = {}
    for name in ("torch", "numpy", "pandas", "transformers", "scikit-learn", "PyYAML", "xgboost"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    manifest = {"format_version": 2 if config.model.vital_only else 1,
                "model_type": "xgboost" if is_xgb else "torch", "vital_only": config.model.vital_only,
                "project_version": __version__, "code_sha256": source_fingerprint(),
                "python": platform.python_version(), "packages": packages,
                "horizon_hours": config.data.horizon, "text_identity": backend.identity if backend else None,
                "preprocessing_version": preprocessor.identifier,
                "calibrator": calibrator.to_dict() if calibrator else None,
                "threshold": threshold_policy["threshold"], "threshold_strategy": threshold_policy["threshold_strategy"],
                "target_recall": threshold_policy["target_recall"], "probability_space": probability_space,
                "threshold_policy": threshold_policy, "vital_encoder": config.model.vital_encoder,
                "training": training_record or {}}
    manifest["files"] = {p.relative_to(path).as_posix(): file_sha256(p) for p in sorted(path.rglob("*")) if p.is_file()}
    manifest["model_version"] = fingerprint(manifest)
    write_json(path / "manifest.json", manifest)
    return manifest


def load_artifact(path, device="auto"):
    path = Path(path).resolve()
    manifest = read_json(path / "manifest.json")
    if manifest.get("format_version") not in (1, 2):
        raise ValueError("Unsupported model artifact version")
    identity = dict(manifest)
    claimed = identity.pop("model_version")
    if fingerprint(identity) != claimed:
        raise ValueError("Artifact manifest checksum mismatch")
    for name, digest in manifest["files"].items():
        file = (path / name).resolve()
        try:
            file.relative_to(path)
        except ValueError as exc:
            raise ValueError(f"Artifact path escapes its directory: {name}") from exc
        if not file.is_file() or file_sha256(file) != digest:
            raise ValueError(f"Artifact integrity error: {name}")
    is_xgb = manifest.get("model_type") == "xgboost"
    required = {"model.ubj" if is_xgb else "weights.pt", "config.json", "contract.json", "preprocessing.json", "normalization.json", "schema.json"}
    if manifest.get("vital_only"):
        required.add("input_layout.json")
    if not required <= manifest["files"].keys():
        raise ValueError("Incomplete artifact manifest")
    cfg = Config.from_dict(read_json(path / "config.json"))
    probability_space = "calibrated" if manifest.get("calibrator") else "raw"
    policy = validate_policy(manifest.get("threshold_policy"), probability_space=probability_space,
                             horizon_hours=cfg.data.horizon)
    if (manifest.get("threshold") != policy["threshold"] or manifest.get("probability_space") != probability_space
            or manifest.get("horizon_hours") != cfg.data.horizon
            or cfg.training.calibration != bool(manifest.get("calibrator"))):
        raise ValueError("Artifact threshold/calibration/horizon metadata mismatch")
    if "threshold.json" not in manifest["files"] or read_json(path / "threshold.json") != policy:
        raise ValueError("Missing/inconsistent saved threshold.json")
    if cfg.model.vital_only != manifest.get("vital_only", False) or is_xgb != (cfg.model.vital_encoder == "xgboost"):
        raise ValueError("Artifact model type/mode mismatch")
    if cfg.model.vital_only and read_json(path / "input_layout.json") != vital_input_layout():
        raise ValueError("Unsupported vital feature order/layout")
    contract = Contract(**read_json(path / "contract.json"))
    params = read_json(path / "preprocessing.json")
    if params.get("version") != ("vital-numeric-v1" if cfg.model.vital_only else "fixed-numeric-v1"):
        raise ValueError("Only fixed-cohort preprocessing is supported")
    preprocessor = (VitalNumericPreprocessor if cfg.model.vital_only else FixedNumericPreprocessor)(**params)
    Contract(**preprocessor.contract).assert_matches(contract)
    if preprocessor.identifier != manifest["preprocessing_version"]:
        raise ValueError("Preprocessing version mismatch")
    if is_xgb:
        from ..models.vital_xgboost import require_xgboost
        model = require_xgboost().Booster()
        model.load_model(path / "model.ubj")
        model.set_param({"device": "cpu", "nthread": cfg.xgboost.nthread})
        if model.num_features() != 2400:
            raise ValueError("XGBoost requires 2400 flattened vital features")
        return model, cfg, contract, preprocessor, manifest
    device = resolve_device(device)
    model = make_risk_model(cfg.model, len(contract.vital_names), len(contract.lab_names),
                            manifest["text_identity"]["dimension"] if manifest["text_identity"] else 0,
                            vital_channels=preprocessor.output_vital_channels)
    state = torch.load(path / "weights.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, cfg, contract, preprocessor, manifest

