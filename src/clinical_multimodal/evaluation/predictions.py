"""Persist per-window probabilities, including explicit invalid input rows."""
import numpy as np
import pandas as pd
from .calibration import sigmoid
from .thresholds import apply_policy


def save_predictions(path, collected, config, calibrator, policy):
    probabilities = calibrator.predict(collected["logits"]) if calibrator else sigmoid(collected["logits"])
    valid = collected["valid"]
    frame = pd.DataFrame({"sample_id": [str(m["sample_id"]) for m in collected["metadata"]],
                          "horizon_hours": config.data.horizon, "target": collected["targets"],
                          "label_valid": collected["label_mask"],
                          "risk": np.where(valid, probabilities, np.nan),
                          "status": np.where(valid, "ok", "invalid_input_no_vitals" if config.model.vital_only
                                             else "invalid_input_no_dynamic_modality")})
    predictions = apply_policy(probabilities, policy, probability_space="calibrated" if calibrator else "raw",
                               horizon_hours=config.data.horizon)
    frame["prediction"] = [int(p) if v else None for p, v in zip(predictions, valid)]
    frame["alarm"] = [bool(p) if v else None for p, v in zip(predictions, valid)]
    frame.to_csv(path, index=False)
