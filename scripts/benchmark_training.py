"""Bounded synthetic GPU benchmark; never opens patient data or trains an artifact."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from clinical_multimodal.config import ModelConfig, TrainingConfig
from clinical_multimodal.models.predictor import VitalRiskModel
from clinical_multimodal.training.trainer import Trainer, seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["tcn", "lstm", "gru", "grud"])
    parser.add_argument("--batches", nargs="+", type=int, default=[16, 64, 128])
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA required")
    if args.steps <= 0 or args.warmup < 0 or any(b <= 0 for b in args.batches):
        parser.error("steps and batches must be positive; warmup must be nonnegative")
    records = []
    for encoder in args.models:
        for size in args.batches:
            gc.collect()
            torch.cuda.empty_cache()
            seed_everything(42, deterministic=not args.fast)
            cfg = TrainingConfig(device="cuda:0", precision="auto", batch_size=size,
                                 deterministic=not args.fast)
            if hasattr(cfg, "allow_tf32"):
                cfg.allow_tf32 = args.fast
            trainer = Trainer(VitalRiskModel(ModelConfig(vital_encoder=encoder, vital_only=True)), cfg, Path("."))
            # Includes pinned CPU -> GPU transfer, all normal model validation,
            # forward/backward/optimizer. Excludes disk, metadata and text encoding.
            batch = {"inputs": {"vitals": torch.randn(size, 120, 20).pin_memory(),
                                "time_mask": torch.ones(size, 120, dtype=torch.bool).pin_memory(),
                                "available": torch.tensor([[True, False, False]] * size).pin_memory()},
                     "target": (torch.arange(size) % 2).float().pin_memory(),
                     "label_mask": torch.ones(size, dtype=torch.bool).pin_memory(), "metadata": []}
            batch["inputs"]["vitals"][:, :, 14:] = (torch.rand(size, 120, 6) > .2).float()
            for _ in range(args.warmup):
                trainer.train_batch(batch)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            for _ in range(args.steps):
                result = trainer.train_batch(batch)
                if result is None or not all(torch.isfinite(torch.tensor(result))):
                    raise RuntimeError("Invalid training step")
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            record = dict(model=encoder, batch_size=size, precision=trainer.precision,
                          deterministic=cfg.deterministic, allow_tf32=getattr(cfg, "allow_tf32", False),
                          samples_per_second=size * args.steps / seconds,
                          step_ms=seconds * 1000 / args.steps,
                          peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20)
            records.append(record)
            print(json.dumps(record), flush=True)
            del trainer, batch
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
        cuda=torch.version.cuda, warmup=args.warmup, steps=args.steps,
        scope="synthetic vital-only training steps; no data loading/text/validation/checkpoints", results=records),
        indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
