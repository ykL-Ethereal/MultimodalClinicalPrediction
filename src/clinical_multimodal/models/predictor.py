import torch
from torch import nn
from .vital_tcn import VitalTCN
from .vital_rnn import VitalRNN
from .vital_grud import VitalGRUD
from .lab_resnet import LabResNet
from .note_encoder import NoteEncoder
from .static_encoder import StaticEncoder
from .fusion import Fusion


class MultimodalRiskModel(nn.Module):
    def __init__(self, n_vitals, n_labs, text_dimension, config, *, vital_channels=None):
        super().__init__()
        channels = 20 if vital_channels is None else vital_channels
        if config.vital_encoder == "tcn":
            self.vital = VitalTCN(n_vitals, config, input_channels=channels)
        elif config.vital_encoder in ("lstm", "gru"):
            self.vital = VitalRNN(config, input_channels=channels)
        elif config.vital_encoder == "grud":
            self.vital = VitalGRUD(config, input_channels=channels)
        else:
            raise ValueError(f"Unknown vital encoder: {config.vital_encoder}")
        self.lab = LabResNet(n_labs, config)
        self.note = NoteEncoder(text_dimension, config.embedding_dim)
        self.static = StaticEncoder(config.static_dim)
        self.fusion = Fusion(config)

    def forward(self, x):
        available = x["available"]
        representations = torch.stack([
            self.vital(x["vitals"], x["time_mask"], available[:, 0]),
            self.lab(x["labs"], available[:, 1]),
            self.note(x["note_chunks"], x["chunk_mask"], x["note_age"], x["note_mask"]),
        ], dim=1)
        logits, weights, used = self.fusion(representations, self.static(x["static"]), available, x["quality"])
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite model output; check input magnitudes and training stability")
        return {"logits": logits, "valid": available.any(dim=1), "used_modalities": used, "gate_weights": weights}

