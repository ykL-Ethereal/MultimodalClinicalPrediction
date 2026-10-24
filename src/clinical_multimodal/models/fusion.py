import torch
from torch import nn
from .pooling import masked_softmax


class Fusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mode = config.fusion
        self.drop_probability = config.modality_dropout
        d, s = config.embedding_dim, config.static_dim
        self.projections = nn.ModuleList([nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d)) for _ in range(3)])
        self.gates = nn.ModuleList([nn.Sequential(nn.Linear(d + s + 3, 64), nn.ReLU(), nn.Linear(64, 1)) for _ in range(3)])
        self.head = nn.Sequential(nn.Linear(d * 3 + s + 3, 128), nn.ReLU(), nn.Dropout(config.dropout), nn.Linear(128, 1))

    def drop_modalities(self, available):
        if not self.training or self.drop_probability == 0:
            return available
        kept = available & (torch.rand_like(available, dtype=torch.float32) >= self.drop_probability)
        empty = available.any(dim=1) & ~kept.any(dim=1)
        scores = torch.rand_like(available, dtype=torch.float32).masked_fill(~available, -1)
        fallback = scores.argmax(dim=1)
        kept[empty, fallback[empty]] = True
        return kept

    def forward(self, representations, static, available, quality):
        used = self.drop_modalities(available)
        quality = quality.masked_fill(~used[..., None], 0)
        z = representations.masked_fill(~used[..., None], 0)
        if self.mode == "gated":
            z = torch.stack([p(z[:, i]) for i, p in enumerate(self.projections)], dim=1)
            z = z.masked_fill(~used[..., None], 0)
            logits = torch.cat([g(torch.cat([z[:, i], static, quality[:, i]], dim=-1))
                                for i, g in enumerate(self.gates)], dim=1)
            weights = masked_softmax(logits, used)
            z = z * weights[..., None]
        else:
            weights = used.to(z.dtype)
        combined = torch.cat([z.flatten(1), static, used.to(z.dtype)], dim=1)
        return self.head(combined).squeeze(-1), weights, used

