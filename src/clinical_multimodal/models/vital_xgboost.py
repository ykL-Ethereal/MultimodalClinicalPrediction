"""XGBoost consumes all 120 x 20 preprocessed values, including observation flags."""
import numpy as np


def require_xgboost():
    try:
        import xgboost
    except ImportError as exc:
        raise ImportError("XGBoost requires xgboost>=3.1,<4; install the project requirements") from exc
    if tuple(int(x) for x in xgboost.__version__.split('.')[:2]) < (3, 1):
        raise ImportError("XGBoost >= 3.1 is required for external-memory quantile training")
    return xgboost


def flatten_vitals(vitals):
    values = np.asarray(vitals, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (120, 20):
        raise ValueError("Expected vital input [N,120,20]")
    if not np.isfinite(values).all():
        raise ValueError("Non-finite vital inputs")
    # Zero is a valid standardized value or mask bit, never a missing sentinel.
    return np.ascontiguousarray(values.reshape(len(values), 2400))


def predict_margins(model, vitals):
    xgb = require_xgboost()
    matrix = xgb.DMatrix(flatten_vitals(vitals), nthread=int(model.attr("prediction_nthread") or 1))
    end = int(model.attr("best_iteration")) + 1 if model.attr("best_iteration") is not None else model.num_boosted_rounds()
    result = model.predict(matrix, output_margin=True, iteration_range=(0, end))
    if not np.isfinite(result).all():
        raise FloatingPointError("Non-finite XGBoost predictions")
    return result
