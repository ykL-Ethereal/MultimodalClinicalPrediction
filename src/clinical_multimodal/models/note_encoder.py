import math
import torch
from torch import nn
from .pooling import AttentionPool


def positions(length, dimension, device, dtype):
    p = torch.arange(length, device=device, dtype=dtype)[:, None]
    frequency = torch.exp(torch.arange(0, dimension, 2, device=device, dtype=dtype) * (-math.log(10000.) / dimension))
    output = torch.zeros(length, dimension, device=device, dtype=dtype)
    output[:, 0::2] = torch.sin(p * frequency)
    output[:, 1::2] = torch.cos(p * frequency[:dimension // 2])
    return output


class NoteEncoder(nn.Module):
    """Trainable block-to-document-to-sample pooling over frozen backbone outputs."""
    def __init__(self, text_dimension, embedding_dimension=128):
        super().__init__()
        self.project = nn.Linear(text_dimension, embedding_dimension)
        self.chunk_pool = AttentionPool(embedding_dimension)
        self.age_project = nn.Sequential(nn.Linear(embedding_dimension + 1, embedding_dimension), nn.Tanh())
        self.document_pool = AttentionPool(embedding_dimension)

    def forward(self, chunks, chunk_mask, note_age, note_mask):
        chunks = chunks.masked_fill(~chunk_mask[..., None], 0)
        z = self.project(chunks)
        z = z + positions(z.shape[2], z.shape[-1], z.device, z.dtype)[None, None]
        documents, _ = self.chunk_pool(z, chunk_mask)
        observed = note_mask & chunk_mask.any(dim=-1)
        age = note_age.masked_fill(~observed, 0)
        documents = self.age_project(torch.cat([documents, age[..., None]], dim=-1))
        out, _ = self.document_pool(documents, observed)
        return out.masked_fill(~observed.any(dim=1, keepdim=True), 0)

