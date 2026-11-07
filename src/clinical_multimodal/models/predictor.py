import torch
from torch import nn
from .vital_tcn import VitalTCN
from .vital_rnn import VitalRNN
from .vital_grud import VitalGRUD
from .lab_resnet import LabResNet
from .note_encoder import NoteEncoder
from .static_encoder import StaticEncoder
from .fusion import Fusion


def make_vital_encoder(config, channels=20):
    if config.vital_encoder == "tcn":
        return VitalTCN(14, config, input_channels=channels)
    if config.vital_encoder in ("lstm", "gru"):
        return VitalRNN(config, input_channels=channels)
    if config.vital_encoder == "grud":
        return VitalGRUD(config, input_channels=channels)
    raise ValueError(f"Unknown vital encoder: {config.vital_encoder}")


class VitalRiskModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vital = make_vital_encoder(config)
        self.head = nn.Linear(config.embedding_dim, 1)

    def forward(self, x):
        valid = x["available"][:, 0]
        logits = self.head(self.vital(x["vitals"], x["time_mask"], valid)).squeeze(-1)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite vital-only model output")
        used = torch.zeros((len(valid), 3), dtype=torch.bool, device=valid.device)
        used[:, 0] = valid
        return {"logits": logits, "valid": valid, "used_modalities": used, "gate_weights": used.float()}


def make_risk_model(config, n_vitals=14, n_labs=0, text_dimension=0, *, vital_channels=20):
    if config.vital_only:
        return VitalRiskModel(config)
    return MultimodalRiskModel(n_vitals, n_labs, text_dimension, config, vital_channels=vital_channels)


class MultimodalRiskModel(nn.Module):
    def __init__(self, n_vitals, n_labs, text_dimension, config, *, vital_channels=None):
        super().__init__()
        channels = 20 if vital_channels is None else vital_channels
        self.vital = make_vital_encoder(config, channels)
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

