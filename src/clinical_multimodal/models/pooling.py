import torch
from torch import nn


def masked_softmax(logits, mask, dim=-1):
    """All-missing slices return exactly zero, with finite backward gradients."""
    masked = logits.masked_fill(~mask, -torch.inf)
    any_valid = mask.any(dim=dim, keepdim=True)
    safe = torch.where(any_valid, masked, torch.zeros_like(masked))
    return torch.softmax(safe, dim=dim).masked_fill(~mask, 0)


class AttentionPool(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dimension, dimension), nn.Tanh(), nn.Linear(dimension, 1))

    def forward(self, values, mask):
        clean = values.masked_fill(~mask[..., None], 0)
        weights = masked_softmax(self.score(clean).squeeze(-1), mask)
        return (clean * weights[..., None]).sum(dim=-2), weights

