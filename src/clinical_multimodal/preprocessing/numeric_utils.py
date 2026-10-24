"""Shared masked numeric operations for the fixed data package."""
import numpy as np


def masked(values, mask):
    return np.where(mask, values, 0).astype(np.float32)


def log_age(values, mask, scale, cap=None):
    x = np.where(mask, values, 0).astype(np.float64)
    if cap is not None:
        x = np.minimum(x, cap)
    return (np.log1p(x) / np.log1p(scale)).astype(np.float32)
