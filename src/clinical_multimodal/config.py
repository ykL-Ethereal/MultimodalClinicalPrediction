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
    device: str = "auto"
    precision: str = "float32"


@dataclass
class ModelConfig:
    vital_encoder: Literal["tcn", "lstm", "gru", "grud", "news2", "xgboost"] = "tcn"
    vital_only: bool = False
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
    threshold_strategy: str = "recall_constraint"
    target_recall: float = 0.80
    threshold_fraction: float = 0.30
    device: str = "auto"
    precision: str = "float32"
    deterministic: bool = True
    num_threads: int = 4
    log_every_steps: int = 100
    output_dir: str = "../runs"


@dataclass
class NEWS2Config:
    rules_file: str = str(Path(__file__).resolve().parents[2] / "configs" / "news2_rules.yaml")
    batch_size: int = 256


@dataclass
class XGBoostConfig:
    num_boost_round: int = 500
    early_stopping_rounds: int = 30
    max_depth: int = 6
    learning_rate: float = 0.05
    min_child_weight: float = 1.0
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    max_bin: int = 256
    batch_size: int = 16384
    # CPU hist with external-memory quantile pages; independent of torch/CUDA settings.
    nthread: int = 4


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    text: TextConfig = field(default_factory=TextConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    news2: NEWS2Config = field(default_factory=NEWS2Config)
    xgboost: XGBoostConfig = field(default_factory=XGBoostConfig)

    def validate(self) -> "Config":
        for section in (self.data, self.text, self.model, self.training, self.news2, self.xgboost):
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
        encoders = {"tcn", "lstm", "gru", "grud", "news2", "xgboost"}
        if self.model.vital_encoder not in encoders or not set(self.training.models) <= encoders:
            raise ValueError("Models must be tcn, lstm, gru, grud, news2 or xgboost")
        if (self.model.vital_encoder == "xgboost" or "xgboost" in self.training.models) and not self.model.vital_only:
            raise ValueError("XGBoost requires --vital_only or model.vital_only: true")
        if len(set(self.training.models)) != len(self.training.models):
            raise ValueError("training.models must not contain duplicates")
        if self.text.backend != "macbert" or self.model.fusion not in ("concat", "gated"):
            raise ValueError("Unknown text backend or fusion")
        for section, names in ((self.model, ("embedding_dim", "static_dim", "tcn_channels", "tcn_blocks",
                                             "kernel_size", "max_time_steps", "lab_hidden", "rnn_hidden", "rnn_layers")),
                               (self.training, ("batch_size", "epochs", "patience", "num_threads", "log_every_steps")),
                               (self.text, ("batch_size",)), (self.news2, ("batch_size",))):
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
        from .evaluation.thresholds import STRATEGIES, validate_threshold
        if self.training.threshold_strategy not in STRATEGIES:
            raise ValueError("Unknown threshold_strategy")
        if not 0 < self.training.target_recall <= 1 or not 0 < self.training.threshold_fraction < 1:
            raise ValueError("target_recall must lie in (0, 1]; threshold_fraction in (0, 1)")
        if self.training.threshold is not None:
            validate_threshold(self.training.threshold)
        if self.training.threshold_strategy == "fixed" and self.training.threshold is None:
            raise ValueError("fixed threshold_strategy requires an explicit threshold")
        if self.training.threshold_strategy != "fixed" and self.training.threshold is not None:
            raise ValueError("Set threshold only with threshold_strategy: fixed")
        if not 0 <= self.training.seed < 2 ** 32:
            raise ValueError("seed must lie in [0, 2**32)")
        xgb = self.xgboost
        if any(getattr(xgb, name) <= 0 for name in ("num_boost_round", "early_stopping_rounds",
                "max_depth", "max_bin", "batch_size", "nthread")) or xgb.max_bin < 2:
            raise ValueError("XGBoost dimensions/iterations must be positive and max_bin >= 2")
        if xgb.learning_rate <= 0 or xgb.min_child_weight < 0 or xgb.reg_lambda < 0:
            raise ValueError("Invalid XGBoost learning rate or regularization")
        if not 0 < xgb.subsample <= 1 or not 0 < xgb.colsample_bytree <= 1:
            raise ValueError("XGBoost sampling fractions must lie in (0, 1]")
        for section in (self.text, self.training):
            if section.precision not in ("auto", "float32", "float16", "bfloat16"):
                raise ValueError("precision must be auto, float32, float16 or bfloat16")
            device = section.device
            if device not in ("auto", "cpu", "cuda") and not (
                    device.startswith("cuda:") and device[5:].isdigit()):
                raise ValueError("device must be auto, cpu, cuda or cuda:<index>")
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> "Config":
        types = {"data": DataConfig, "text": TextConfig, "model": ModelConfig, "training": TrainingConfig,
                 "news2": NEWS2Config, "xgboost": XGBoostConfig}
        if not isinstance(raw, dict) or set(raw) - types.keys():
            raise ValueError("Unknown configuration sections")
        try:
            return cls(**{k: types[k](**v) for k, v in raw.items()}).validate()
        except TypeError as exc:
            raise ValueError(f"Invalid configuration field: {exc}") from exc


def load_config(path: str | Path, *, vital_only=None, models=None) -> Config:
    import yaml
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    # Apply CLI overrides before cross-field validation (e.g. xgboost + --vital_only).
    if isinstance(raw, dict):
        if vital_only is not None:
            raw.setdefault("model", {})["vital_only"] = vital_only
        if models:
            raw.setdefault("training", {})["models"] = models
            raw.setdefault("model", {})["vital_encoder"] = models[0]
    cfg = Config.from_dict(raw)
    for section, name in ((cfg.data, "root"), (cfg.text, "cache_dir"), (cfg.training, "output_dir"),
                          (cfg.news2, "rules_file")):
        p = Path(getattr(section, name)).expanduser()
        setattr(section, name, str(p if p.is_absolute() else (path.parent / p).resolve()))
    local_model = Path(cfg.text.model_name).expanduser()
    if not local_model.is_absolute() and (path.parent / local_model).is_dir():
        cfg.text.model_name = str((path.parent / local_model).resolve())
    return cfg

