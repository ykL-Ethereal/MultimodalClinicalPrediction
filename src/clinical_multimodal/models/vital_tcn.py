"""Causal dilated residual convolutions inspired by locuslab/TCN.

Explicitly left-pad, enforce stage masks after every convolution, and normalize
only across channels at each time position.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .vital_rnn import pool_sequence, validate_sequence


class CausalConv1d(nn.Conv1d):
    def forward(self, x):
        padding = self.dilation[0] * (self.kernel_size[0] - 1)
        return super().forward(F.pad(x, (padding, 0)))


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, channels, kernel_size, dilation, dropout):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, channels, kernel_size, dilation=dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation=dilation)
        self.norm1, self.norm2 = nn.LayerNorm(channels), nn.LayerNorm(channels)
        self.residual = nn.Conv1d(in_channels, channels, 1) if in_channels != channels else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = x.masked_fill(~mask[:, None], 0)
        z = self.conv1(x).masked_fill(~mask[:, None], 0)
        z = self.dropout(F.relu(self.norm1(z.transpose(1, 2)).transpose(1, 2)))
        z = z.masked_fill(~mask[:, None], 0)
        z = self.conv2(z).masked_fill(~mask[:, None], 0)
        z = self.dropout(F.relu(self.norm2(z.transpose(1, 2)).transpose(1, 2)))
        return F.relu(z + self.residual(x)).masked_fill(~mask[:, None], 0)


class VitalTCN(nn.Module):
    def __init__(self, n_vitals, config, *, input_channels=None):
        super().__init__()
        self.max_time_steps = config.max_time_steps
        channels = 20 if input_channels is None else input_channels
        self.blocks = nn.ModuleList([TemporalBlock(channels if i == 0 else config.tcn_channels,
                                    config.tcn_channels, config.kernel_size, 2 ** i, config.dropout)
                                    for i in range(config.tcn_blocks)])
        self.project = nn.Linear(config.tcn_channels * 2, config.embedding_dim)

    def sequence(self, x, time_mask):
        validate_sequence(x, time_mask, self.max_time_steps)
        z = x.transpose(1, 2)
        for block in self.blocks:
            z = block(z, time_mask)
        return z.transpose(1, 2)

    def forward(self, x, time_mask, available):
        z = self.sequence(x, time_mask)
        return pool_sequence(z, time_mask, available, self.project)

