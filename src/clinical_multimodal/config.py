"""Validated configuration. Relative paths are relative to the configuration file."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Literal
import math


@dataclass
class DataConfig:
    root: str = "../data/Mutimodel_data/固定训练数据"
    horizon: int = 6


@dataclass
class TextConfig:
    backend: Literal["macbert"] = "macbert"
    model_name: str = "hfl/chinese-macbert-base"
    revision: str = "main"
    local_files_only: bool = False
    max_tokens: int = 512
    batch_size: int = 16
    cache_dir: str = "../runs/text_cache"
    device: str = "cpu"


@dataclass
class ModelConfig:
    vital_encoder: Literal["tcn", "lstm", "gru", "grud"] = "tcn"
    rnn_hidden: int = 64
    rnn_layers: int = 2
    fusion: Literal["concat", "gated"] = "gated"
    embedding_dim: int = 128
    static_dim: int = 32
    tcn_channels: int = 64
    tcn_blocks: int = 5
    kernel_size: int = 3
    max_time_steps: int = 120
    lab_hidden: int = 256
    dropout: float = 0.1
    modality_dropout: float = 0.1

    @property
    def receptive_field(self) -> int:
        return 1 + 2 * (self.kernel_size - 1) * (2 ** self.tcn_blocks - 1)


@dataclass
class TrainingConfig:
    models: list[str] = field(default_factory=lambda: ["tcn", "lstm", "gru"])
    seed: int = 20261010
    batch_size: int = 16
    epochs: int = 50
    patience: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    class_weight: bool = False
    calibration: bool = False
    calibration_fraction: float = 0.5
    threshold: float | None = None
    device: str = "cpu"
    num_threads: int = 4
    log_every_steps: int = 100
    output_dir: str = "../runs"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    text: TextConfig = field(default_factory=TextConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def validate(self) -> "Config":
        for section in (self.data, self.text, self.model, self.training):
            defaults = type(section)()
            for f in fields(section):
                value, default = getattr(section, f.name), getattr(defaults, f.name)
                if isinstance(default, bool):
                    good = type(value) is bool
                elif isinstance(default, int):
                    good = type(value) is int
                elif isinstance(default, float):
                    good = type(value) in (int, float) and math.isfinite(value)
                elif isinstance(default, str):
                    good = isinstance(value, str) and bool(value)
                elif isinstance(default, list):
                    good = isinstance(value, list) and bool(value) and all(isinstance(x, str) for x in value)
                else:
                    good = value is None or (type(value) in (int, float) and math.isfinite(value))
                if not good:
                    raise ValueError(f"Invalid type/value: {type(section).__name__}.{f.name}")
        if self.data.horizon not in (1, 2, 3, 6):
            raise ValueError("horizon must be 1, 2, 3 or 6 hours")
        encoders = {"tcn", "lstm", "gru", "grud"}
        if self.model.vital_encoder not in encoders or not set(self.training.models) <= encoders:
            raise ValueError("Vital models must be tcn, lstm, gru or grud")
        if len(set(self.training.models)) != len(self.training.models):
            raise ValueError("training.models must not contain duplicates")
        if self.text.backend != "macbert" or self.model.fusion not in ("concat", "gated"):
            raise ValueError("Unknown text backend or fusion")
        for section, names in ((self.model, ("embedding_dim", "static_dim", "tcn_channels", "tcn_blocks",
                                             "kernel_size", "max_time_steps", "lab_hidden", "rnn_hidden", "rnn_layers")),
                               (self.training, ("batch_size", "epochs", "patience", "num_threads", "log_every_steps")),
                               (self.text, ("batch_size",))):
            if any(getattr(section, name) <= 0 for name in names):
                raise ValueError("Dimensions and iteration counts must be positive")
        if not 4 <= self.text.max_tokens <= 512:
            raise ValueError("max_tokens must lie in [4, 512]")
        if self.model.vital_encoder == "tcn" and self.model.receptive_field < self.model.max_time_steps:
            raise ValueError("TCN receptive field is shorter than max_time_steps")
        if any(not 0 <= p < 1 for p in (self.model.dropout, self.model.modality_dropout)):
            raise ValueError("Dropout probabilities must lie in [0, 1)")
        if self.training.learning_rate <= 0 or self.training.weight_decay < 0 or self.training.gradient_clip <= 0:
            raise ValueError("Invalid optimizer parameters")
        if not 0 < self.training.calibration_fraction < 1:
            raise ValueError("calibration_fraction must lie in (0, 1)")
        if self.training.threshold is not None and not 0 < self.training.threshold < 1:
            raise ValueError("threshold must lie in (0, 1)")
        if not 0 <= self.training.seed < 2 ** 32:
            raise ValueError("seed must lie in [0, 2**32)")
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> "Config":
        types = {"data": DataConfig, "text": TextConfig, "model": ModelConfig, "training": TrainingConfig}
        if not isinstance(raw, dict) or set(raw) - types.keys():
            raise ValueError("Unknown configuration sections")
        try:
            return cls(**{k: types[k](**v) for k, v in raw.items()}).validate()
        except TypeError as exc:
            raise ValueError(f"Invalid configuration field: {exc}") from exc


def load_config(path: str | Path) -> Config:
    import yaml
    path = Path(path).resolve()
    cfg = Config.from_dict(yaml.safe_load(path.read_text(encoding="utf-8")))
    for section, name in ((cfg.data, "root"), (cfg.text, "cache_dir"), (cfg.training, "output_dir")):
        p = Path(getattr(section, name)).expanduser()
        setattr(section, name, str(p if p.is_absolute() else (path.parent / p).resolve()))
    local_model = Path(cfg.text.model_name).expanduser()
    if not local_model.is_absolute() and (path.parent / local_model).is_dir():
        cfg.text.model_name = str((path.parent / local_model).resolve())
    return cfg

