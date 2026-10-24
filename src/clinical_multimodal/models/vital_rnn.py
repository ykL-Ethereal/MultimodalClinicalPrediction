"""Unidirectional encoders using PyTorch LSTM/GRU and packed sequences.

https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html
https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html
Time masks denote right padding, not feature-level missing observations.
"""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def validate_sequence(x, time_mask, max_time_steps):
    if x.ndim != 3 or time_mask.shape != x.shape[:2] or time_mask.dtype != torch.bool:
        raise ValueError("Expected vitals [B,T,C] and boolean time_mask [B,T]")
    if not 0 < x.shape[1] <= max_time_steps:
        raise ValueError("Vital sequence length is outside the trained contract")
    if (time_mask[:, 1:] & ~time_mask[:, :-1]).any():
        raise ValueError("time_mask must use right padding; missing features use observation masks")


def pool_sequence(z, time_mask, available, project):
    z = z.masked_fill(~time_mask[..., None], 0)
    lengths = time_mask.sum(dim=1)
    last = z[torch.arange(len(z), device=z.device), (lengths - 1).clamp(min=0)]
    mean = z.sum(dim=1) / lengths[:, None].clamp(min=1)
    out = project(torch.cat([last, mean], dim=-1))
    return out.masked_fill(~(available & (lengths > 0))[:, None], 0)


class VitalRNN(nn.Module):
    def __init__(self, config, *, input_channels):
        super().__init__()
        self.max_time_steps = config.max_time_steps
        recurrent = {"lstm": nn.LSTM, "gru": nn.GRU}[config.vital_encoder]
        self.rnn = recurrent(input_channels, config.rnn_hidden, config.rnn_layers,
                             batch_first=True, bidirectional=False,
                             dropout=config.dropout if config.rnn_layers > 1 else 0.)
        self.project = nn.Sequential(nn.Dropout(config.dropout),
                                     nn.Linear(config.rnn_hidden * 2, config.embedding_dim))

    def sequence(self, x, time_mask):
        validate_sequence(x, time_mask, self.max_time_steps)
        clean = x.masked_fill(~time_mask[..., None], 0)
        # A dummy zero step makes empty rows packable; their outputs are masked.
        packed = pack_padded_sequence(clean, time_mask.sum(dim=1).clamp(min=1).cpu(),
                                      batch_first=True, enforce_sorted=False)
        encoded, _ = self.rnn(packed)
        z, _ = pad_packed_sequence(encoded, batch_first=True, total_length=x.shape[1])
        return z.masked_fill(~time_mask[..., None], 0)

    def forward(self, x, time_mask, available):
        return pool_sequence(self.sequence(x, time_mask), time_mask, available, self.project)
