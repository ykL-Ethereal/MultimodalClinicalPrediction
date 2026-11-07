"""Portable NEWS2 artifact and batched predictions; no neural-model dependencies."""
from pathlib import Path

import numpy as np
import yaml

from ..data.schema import Contract
from ..io import file_sha256, fingerprint, read_json, write_json
from ..models.news2 import NEWS2Model, load_rules


class NEWS2Predictor:
    def __init__(self, artifact):
        self.path = Path(artifact)
        self.manifest = read_json(self.path / "manifest.json")
        identity = dict(self.manifest)
        claimed = identity.pop("manifest_sha256", None)
        if fingerprint(identity) != claimed:
            raise ValueError("NEWS2 manifest integrity check failed")
        if self.manifest.get("model_type") != "news2":
            raise ValueError("Not a NEWS2 artifact")
        rules_path = self.path / "news2_rules.yaml"
        if file_sha256(rules_path) != self.manifest["rules_sha256"]:
            raise ValueError("NEWS2 rules integrity check failed")
        self.model = NEWS2Model(load_rules(rules_path))
        if self.model.version != self.manifest["model_version"]:
            raise ValueError("NEWS2 model version mismatch")
        self.contract = Contract(**self.manifest["contract"])
        self.horizon = self.manifest["horizon_hours"]

    @classmethod
    def save(cls, path, rules, contract, horizon):
        model = NEWS2Model(rules)
        path = Path(path)
        path.mkdir(parents=True, exist_ok=False)
        rules_path = path / "news2_rules.yaml"
        rules_path.write_text(yaml.safe_dump(model.rules, allow_unicode=True, sort_keys=False), encoding="utf-8")
        manifest = {"model_type": "news2", "variant": "reduced",
            "model_version": model.version, "horizon_hours": horizon, "contract": contract.to_dict(),
            "rules_sha256": file_sha256(rules_path), "probability_available": False}
        manifest["manifest_sha256"] = fingerprint(manifest)
        write_json(path / "manifest.json", manifest)
        return cls(path)

    def iter_predict(self, bundle, *, rows=None, batch_size=256):
        self.contract.assert_matches(bundle.contract)
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        rows = np.arange(len(bundle.metadata)) if rows is None else np.asarray(rows)
        if rows.ndim != 1 or (rows.size and (rows.dtype.kind not in "iu" or (rows < 0).any()
                                            or (rows >= len(bundle.metadata)).any())):
            raise ValueError("Invalid NEWS2 inference row indices")
        rows = rows.astype(np.int64)
        for start in range(0, len(rows), batch_size):
            selected = rows[start:start + batch_size]
            shard_ids = bundle.shard_ids(selected)
            # Copy only vitals, not the large lab tensors or text fields.
            vitals = np.empty((len(selected), bundle.schema["history_minutes"], len(self.contract.vital_names)), dtype=float)
            for sid in np.unique(shard_ids):
                dest = np.flatnonzero(shard_ids == sid)
                vitals[dest] = bundle.shard(sid)["vital_features"][selected[dest] - bundle.starts[sid]]
            scored = self.model.score(vitals, self.contract.vital_names)
            sample_ids = bundle.metadata.iloc[selected]["sample_id"].tolist()
            results = []
            for i, sample_id in enumerate(sample_ids):
                valid = bool(scored["valid"][i])
                components, missing = {}, []
                for name, component in scored["components"].items():
                    present = component["scores"][i] >= 0
                    if not present:
                        missing.append(name)
                    components[name] = {"value": float(component["values"][i]) if present else None,
                        "score": int(component["scores"][i]) if present else None,
                        "source": str(component["sources"][i]) if present else None,
                        "time_index": int(component["time_indices"][i]) if present else None,
                        "unit": component["unit"]}
                results.append({"sample_id": str(sample_id),
                    "horizon_hours": self.horizon, "score": int(scored["scores"][i]) if valid else None,
                    "raw_score": int(scored["scores"][i]) if valid else None,
                    "max_score": int(scored["max_scores"][i]),
                    "normalized_score": float(scored["normalized_scores"][i]) if valid else None,
                    "risk": None, "alarm": bool(scored["scores"][i] >= self.model.rules["alarm_threshold"]) if valid else None,
                    "threshold": self.model.rules["alarm_threshold"], "components": components,
                    "missing_parameters": missing, "omitted_parameters": self.model.rules["omitted_parameters"],
                    "status": ("partial_input" if missing else "ok") if valid else "invalid_input_missing_parameters",
                    "variant": "reduced",
                    "model_version": self.model.version, "purpose": "research_prediction"})
            yield results

    def predict(self, bundle, *, rows=None, batch_size=256):
        return [item for batch in self.iter_predict(bundle, rows=rows, batch_size=batch_size) for item in batch]
