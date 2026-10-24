"""Self-contained, integrity-checked inference artifact; weights-only deserialization."""
from __future__ import annotations

from pathlib import Path
import platform
import importlib.metadata
import torch

from .. import __version__
from ..config import Config
from ..data.schema import Contract
from ..io import file_sha256, fingerprint, read_json, write_json
from ..models.predictor import MultimodalRiskModel
from ..preprocessing.fixed_numeric import FixedNumericPreprocessor


def source_fingerprint():
    source = Path(__file__).resolve().parents[1]
    return fingerprint({str(p.relative_to(source)): file_sha256(p) for p in sorted(source.rglob("*.py"))})


def save_artifact(path, model, config, bundle, preprocessor, backend, *, calibrator=None, training_record=None):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Artifact already exists: {path}")
    path.mkdir(parents=True)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, path / "weights.pt")
    write_json(path / "config.json", config.to_dict())
    write_json(path / "contract.json", bundle.contract.to_dict())
    write_json(path / "normalization.json", bundle.normalization)
    write_json(path / "schema.json", bundle.schema)
    write_json(path / "preprocessing.json", preprocessor.to_dict())
    backend.export(path / "text_backbone")
    packages = {}
    for name in ("torch", "numpy", "pandas", "transformers", "scikit-learn", "PyYAML"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    manifest = {"format_version": 1, "project_version": __version__, "code_sha256": source_fingerprint(),
                "python": platform.python_version(), "packages": packages,
                "horizon_hours": config.data.horizon, "text_identity": backend.identity,
                "preprocessing_version": preprocessor.identifier,
                "calibrator": calibrator.to_dict() if calibrator else None,
                "threshold": config.training.threshold, "vital_encoder": config.model.vital_encoder,
                "training": training_record or {}}
    manifest["files"] = {p.relative_to(path).as_posix(): file_sha256(p) for p in sorted(path.rglob("*")) if p.is_file()}
    manifest["model_version"] = fingerprint(manifest)
    write_json(path / "manifest.json", manifest)
    return manifest


def load_artifact(path, device="cpu"):
    path = Path(path).resolve()
    manifest = read_json(path / "manifest.json")
    if manifest.get("format_version") != 1:
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
    required = {"weights.pt", "config.json", "contract.json", "preprocessing.json", "normalization.json", "schema.json"}
    if not required <= manifest["files"].keys():
        raise ValueError("Incomplete artifact manifest")
    cfg = Config.from_dict(read_json(path / "config.json"))
    contract = Contract(**read_json(path / "contract.json"))
    params = read_json(path / "preprocessing.json")
    if params.get("version") != "fixed-numeric-v1":
        raise ValueError("Only fixed-cohort preprocessing is supported")
    preprocessor = FixedNumericPreprocessor(**params)
    Contract(**preprocessor.contract).assert_matches(contract)
    if preprocessor.identifier != manifest["preprocessing_version"]:
        raise ValueError("Preprocessing version mismatch")
    model = MultimodalRiskModel(len(contract.vital_names), len(contract.lab_names),
                               manifest["text_identity"]["dimension"], cfg.model,
                               vital_channels=getattr(preprocessor, "output_vital_channels", None))
    state = torch.load(path / "weights.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, cfg, contract, preprocessor, manifest

