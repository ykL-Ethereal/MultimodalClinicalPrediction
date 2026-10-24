"""Residual MLP inspired by rtdl-revisiting-models, with batch-independent LayerNorm."""
from torch import nn


class ResidualMLPBlock(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.layers = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.ReLU(),
                                    nn.Dropout(dropout), nn.Linear(width, width), nn.Dropout(dropout))

    def forward(self, x):
        return x + self.layers(x)


class LabResNet(nn.Module):
    def __init__(self, n_labs, config):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(n_labs * 6, config.lab_hidden),
                                     ResidualMLPBlock(config.lab_hidden, config.dropout),
                                     ResidualMLPBlock(config.lab_hidden, config.dropout),
                                     nn.LayerNorm(config.lab_hidden), nn.ReLU(),
                                     nn.Linear(config.lab_hidden, config.embedding_dim))

    def forward(self, x, available):
        x = x.masked_fill(~available[:, None], 0)
        return self.network(x).masked_fill(~available[:, None], 0)

