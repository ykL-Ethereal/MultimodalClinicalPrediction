"""GRU-D equations (10)-(16), Che et al., Scientific Reports 2018.

https://www.nature.com/articles/s41598-018-24271-9
Uses six physical values, actual observation masks and within-window minute
intervals. Means are zero after training-only standardization. History before
the window is unavailable: initial previous values use that training mean.
The paper's reset-before-matrix candidate differs from torch.nn.GRUCell, so
only the decay cell is implemented here with standard torch.nn layers.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .vital_rnn import pool_sequence, validate_sequence


class GRUDCell(nn.Module):
    def __init__(self, features, hidden):
        super().__init__()
        self.input_decay_weight = nn.Parameter(torch.full((features,), 0.1))
        self.input_decay_bias = nn.Parameter(torch.zeros(features))
        self.hidden_decay = nn.Linear(features, hidden)
        self.gates = nn.Linear(features * 2 + hidden, hidden * 2)
        self.candidate = nn.Linear(features * 2 + hidden, hidden)
        nn.init.constant_(self.hidden_decay.weight, 0.1 / features)
        nn.init.zeros_(self.hidden_decay.bias)

    def forward(self, values, observed, delta, previous, h):
        gamma_x = torch.exp(-F.relu(delta * self.input_decay_weight + self.input_decay_bias))
        gamma_h = torch.exp(-F.relu(self.hidden_decay(delta)))
        imputed = torch.where(observed, values, gamma_x * previous)
        decayed = gamma_h * h
        mask = observed.to(values.dtype)
        reset, update = torch.sigmoid(self.gates(torch.cat([imputed, mask, decayed], dim=-1))).chunk(2, dim=-1)
        candidate = torch.tanh(self.candidate(torch.cat([imputed, mask, reset * decayed], dim=-1)))
        return (1 - update) * decayed + update * candidate


class VitalGRUD(nn.Module):
    def __init__(self, config, *, input_channels):
        super().__init__()
        if input_channels != 20:
            raise ValueError("GRU-D requires the fixed cohort's 20 vital channels")
        self.max_time_steps = config.max_time_steps
        self.hidden = config.rnn_hidden
        self.cell = GRUDCell(6, self.hidden)
        self.project = nn.Sequential(nn.Dropout(config.dropout),
                                     nn.Linear(self.hidden * 2, config.embedding_dim))
        self.register_buffer("value_indices", torch.tensor([0, 1, 2, 3, 8, 11]))

    def sequence(self, x, time_mask):
        validate_sequence(x, time_mask, self.max_time_steps)
        x = x.masked_fill(~time_mask[..., None], 0)
        values = x.index_select(-1, self.value_indices)
        observed = x[..., 14:20] > 0
        h = x.new_zeros(len(x), self.hidden)
        previous = x.new_zeros(len(x), 6)
        delta = torch.zeros_like(previous)
        was_observed = torch.zeros_like(previous, dtype=torch.bool)
        outputs = []
        for t in range(x.shape[1]):
            valid = time_mask[:, t, None]
            if t:
                delta = torch.where(was_observed, torch.ones_like(delta), delta + 1)
            step = self.cell(values[:, t], observed[:, t], delta, previous, h)
            h = torch.where(valid, step, h)
            previous = torch.where(valid & observed[:, t], values[:, t], previous)
            was_observed = observed[:, t]
            outputs.append(h.masked_fill(~valid, 0))
        return torch.stack(outputs, dim=1)

    def forward(self, x, time_mask, available):
        return pool_sequence(self.sequence(x, time_mask), time_mask, available, self.project)
