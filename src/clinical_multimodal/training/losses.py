import torch
from torch.nn import functional as F


def masked_bce(logits, targets, label_mask, *, valid=None, pos_weight=None, weights=None):
    selected = label_mask if valid is None else label_mask & valid
    if not selected.any():
        return None
    y = targets[selected]
    if not torch.isfinite(y).all() or not ((y == 0) | (y == 1)).all():
        raise ValueError("Valid labels must be binary")
    losses = F.binary_cross_entropy_with_logits(logits[selected], y, reduction="none", pos_weight=pos_weight)
    if weights is None:
        return losses.mean()
    w = weights[selected]
    if not torch.isfinite(w).all() or (w < 0).any():
        raise ValueError("Sample weights must be finite and nonnegative")
    if w.sum() == 0:
        return None
    return (losses * w).sum() / w.sum()


def training_pos_weight(labels, mask):
    y = labels[mask]
    positive, negative = int((y == 1).sum()), int((y == 0).sum())
    return float(negative / positive) if positive and negative else None

