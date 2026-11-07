"""Configuration-driven reduced NEWS2 on raw, masked physiological values.

All clinical bands, points, source choices and alarm thresholds live in YAML.
This module has no torch, language-model or fitted-preprocessor dependency.
"""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path

import numpy as np
import yaml

from ..io import fingerprint


def load_rules(path):
    rules = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    validate_rules(rules)
    return rules


def validate_rules(rules):
    required = {"version", "source", "omitted_parameters", "time_selection", "missing_policy",
                "alarm_threshold", "parameters"}
    optional = {"interval_closed", "normalization"}
    if not isinstance(rules, dict) or not required <= set(rules) or set(rules) - required - optional:
        raise ValueError("Invalid NEWS2 rule fields")
    if not isinstance(rules["version"], str) or not rules["version"] or not isinstance(rules["source"], dict):
        raise ValueError("NEWS2 rules require version and source metadata")
    if rules["time_selection"] != "latest_available" or rules["missing_policy"] not in ("require_all", "skip"):
        raise ValueError("Unsupported NEWS2 time selection or missing policy")
    if rules.get("interval_closed", "lower") not in ("lower", "upper"):
        raise ValueError("Unsupported NEWS2 interval boundary policy")
    if rules.get("normalization", "available_max") != "available_max":
        raise ValueError("Unsupported NEWS2 normalization")
    omitted = rules["omitted_parameters"]
    if not isinstance(omitted, list) or not all(isinstance(x, str) for x in omitted):
        raise ValueError("Invalid NEWS2 omitted_parameters")
    parameters = rules["parameters"]
    if not isinstance(parameters, dict) or not parameters or set(parameters) & set(omitted):
        raise ValueError("NEWS2 requires nonempty retained parameters disjoint from omissions")
    maximum = 0
    for name, spec in parameters.items():
        required_fields = {"unit", "sources", "cutpoints", "scores"}
        if (not isinstance(name, str) or not isinstance(spec, dict) or not required_fields <= set(spec)
                or set(spec) - required_fields - {"max_score"}):
            raise ValueError("Invalid NEWS2 parameter fields")
        if not isinstance(spec["unit"], str) or not spec["unit"]:
            raise ValueError(f"Invalid NEWS2 unit: {name}")
        cuts, scores = spec["cutpoints"], spec["scores"]
        if (not isinstance(cuts, list) or not all(type(x) in (int, float) and math.isfinite(x) for x in cuts)
                or any(a >= b for a, b in zip(cuts, cuts[1:]))):
            raise ValueError(f"NEWS2 cutpoints must be finite and strictly increasing: {name}")
        if (not isinstance(scores, list) or len(scores) != len(cuts) + 1
                or not all(type(x) is int and x >= 0 for x in scores)):
            raise ValueError(f"NEWS2 scores must be nonnegative integers, one per interval: {name}")
        max_score = spec.get("max_score", max(scores))
        if type(max_score) is not int or max_score <= 0 or max_score < max(scores):
            raise ValueError(f"NEWS2 max_score must be positive and cover every interval score: {name}")
        if not isinstance(spec["sources"], list) or not spec["sources"]:
            raise ValueError(f"NEWS2 sources must be nonempty: {name}")
        for source in spec["sources"]:
            if (not isinstance(source, dict) or set(source) != {"value", "mask", "available_when"}
                    or not all(isinstance(source[k], str) and source[k] for k in ("value", "mask"))
                    or type(source["available_when"]) is not int or source["available_when"] not in (0, 1)):
                raise ValueError(f"Invalid NEWS2 source: {name}")
        maximum += max_score
    threshold = rules["alarm_threshold"]
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= maximum:
        raise ValueError("NEWS2 alarm_threshold must be finite and within the score range")


class NEWS2Model:
    def __init__(self, rules):
        validate_rules(rules)
        self.rules = deepcopy(rules)
        self.version = fingerprint(self.rules)

    def score(self, vital_features, feature_names):
        """Return totals and component diagnostics without filling missing values.

        Each source's last finite masked value is used, in configured source priority.
        The blood-pressure availability mask permits existing historical last-values.
        """
        v = np.asarray(vital_features)
        if (v.ndim != 3 or v.shape[2] != len(feature_names) or v.shape[1] == 0
                or len(set(feature_names)) != len(feature_names)):
            raise ValueError("Invalid NEWS2 vital feature shape/order")
        indices = {name: i for i, name in enumerate(feature_names)}
        n = len(v)
        totals, complete, components = np.zeros(n, dtype=np.int64), np.ones(n, bool), {}
        maximums = np.zeros(n, dtype=np.int64)
        # Old artifact snapshots use lower-inclusive cutpoints; v2 specifies upper.
        side = "left" if self.rules.get("interval_closed", "lower") == "upper" else "right"
        for name, spec in self.rules["parameters"].items():
            values = np.full(n, np.nan)
            source_names = np.full(n, "", dtype=object)
            minutes = np.full(n, -1, dtype=np.int64)
            configured_sources = [s for s in spec["sources"] if s["value"] in indices and s["mask"] in indices]
            if not configured_sources:
                raise ValueError(f"No configured NEWS2 source available for {name}; update the rules file")
            for source in configured_sources:
                raw = v[:, :, indices[source["value"]]]
                mask = v[:, :, indices[source["mask"]]]
                if not np.isin(mask, [0, 1]).all():
                    raise ValueError(f"Invalid NEWS2 mask: {source['mask']}")
                usable = np.isfinite(raw) & (mask == source["available_when"])
                last = np.where(usable, np.arange(v.shape[1]), -1).max(axis=1)
                selected = np.flatnonzero(np.isnan(values) & (last >= 0))
                values[selected] = raw[selected, last[selected]]
                source_names[selected] = source["value"]
                minutes[selected] = last[selected]
            present = np.isfinite(values)
            points = np.full(n, -1, dtype=np.int64)
            points[present] = np.asarray(spec["scores"])[np.searchsorted(spec["cutpoints"], values[present], side=side)]
            totals[present] += points[present]
            maximums[present] += spec.get("max_score", max(spec["scores"]))
            complete &= present
            components[name] = {"values": values, "scores": points, "sources": source_names,
                                "time_indices": minutes, "unit": spec["unit"]}
        valid = maximums > 0 if self.rules["missing_policy"] == "skip" else complete
        normalized = np.full(n, np.nan)
        np.divide(totals, maximums, out=normalized, where=valid & (maximums > 0))
        return {"scores": totals, "max_scores": maximums, "normalized_scores": normalized,
                "valid": valid, "components": components}
