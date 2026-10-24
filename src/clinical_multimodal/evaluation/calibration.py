from dataclasses import asdict, dataclass
import numpy as np
from sklearn.linear_model import LogisticRegression


def sigmoid(logits):
    x = np.asarray(logits, dtype=np.float64)
    z = np.exp(-np.abs(x))
    return np.where(x >= 0, 1 / (1 + z), z / (1 + z))


@dataclass
class PlattCalibrator:
    slope: float
    intercept: float

    @classmethod
    def fit(cls, logits, labels):
        x, y = np.asarray(logits, dtype=float), np.asarray(labels)
        if x.ndim != 1 or x.shape != y.shape or not np.isfinite(x).all() or set(np.unique(y)) != {0, 1}:
            raise ValueError("Calibration requires finite logits and both classes in a held-out set")
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000).fit(x[:, None], y)
        if model.n_iter_[0] >= 2000:
            raise RuntimeError("Probability calibration failed to converge")
        return cls(float(model.coef_[0, 0]), float(model.intercept_[0]))

    def predict(self, logits):
        return sigmoid(self.slope * np.asarray(logits) + self.intercept)

    def to_dict(self):
        return asdict(self)

