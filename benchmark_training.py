"""Measure neural throughput on Train-only inputs; does not select scientific parameters."""
import argparse
from copy import deepcopy
import gc
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from clinical_multimodal.config import load_config
from clinical_multimodal.data.fixed_adapter import FixedCohortBundle
from clinical_multimodal.data.vital_cache import CachedVitalDataset
from clinical_multimodal.io import write_json
from clinical_multimodal.pipelines.patient_cv import configure_device, model_for
from clinical_multimodal.preprocessing.vital_numeric import VitalNumericPreprocessor
from clinical_multimodal.training.trainer import Trainer, make_loader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/patient_cv_30min.yaml"))
    parser.add_argument("--models", nargs="+", default=["gru", "tcn", "lstm", "grud"])
    parser.add_argument("--batches", nargs="+", type=int, default=[128, 256, 512])
    parser.add_argument("--workers", nargs="+", type=int, default=[0, 2])
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("runs/benchmark_30min.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new benchmark output path")
    cfg = load_config(args.config)
    configure_device(cfg)
    if not cfg.training.device.startswith("cuda"):
        raise ValueError("CUDA is required for this hardware benchmark")
    bundle = FixedCohortBundle(cfg.data.root, vital_only=True)
    rng = np.random.default_rng(cfg.training.seed)
    rows = np.sort(rng.choice(bundle.rows("train"), min(8192, len(bundle.rows("train"))), replace=False))
    processor = VitalNumericPreprocessor.fit(bundle, rows, device=cfg.training.device)
    ds = CachedVitalDataset(bundle, rows, processor, cfg.data.horizon, cfg.training.numeric_cache_dir, defer_numeric=True)
    results = []
    for name in args.models:
        for batch_size in args.batches:
            for workers in args.workers:
                current = deepcopy(cfg)
                current.model.vital_encoder = name
                current.training.batch_size = batch_size
                current.training.loader_workers = workers
                current.training.sampling = "uniform_budget"
                current.training.steps_per_epoch = args.steps + 3
                configure_device(current)
                trainer = loader = iterator = model = None
                try:
                    model = model_for(current, bundle, processor)
                    trainer = Trainer(model, current.training, args.output.parent)
                    loader = make_loader(ds, batch_size, shuffle=True, seed=42, device=trainer.device, config=current.training)
                    iterator = iter(loader)
                    for _ in range(3):
                        trainer.train_batch(next(iterator))
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    started = time.perf_counter()
                    for _ in range(args.steps):
                        trainer.train_batch(next(iterator))
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - started
                    row = {"model": name, "batch_size": batch_size, "workers": workers, "status": "ok",
                           "samples_per_second": batch_size * args.steps / elapsed,
                           "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                           "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
                except torch.cuda.OutOfMemoryError:
                    row = {"model": name, "batch_size": batch_size, "workers": workers, "status": "cuda_oom"}
                finally:
                    del iterator, loader, trainer, model
                    gc.collect()
                    torch.cuda.empty_cache()
                results.append(row)
                print(row, flush=True)
                write_json(args.output, {"gpu": torch.cuda.get_device_name(),
                    "total_gpu_bytes": torch.cuda.get_device_properties(0).total_memory,
                    "precision": cfg.training.precision, "deterministic": cfg.training.deterministic,
                    "train_only": True, "warmup_steps": 3, "measured_steps": args.steps,
                    "benchmark_models_discarded": True, "results": results})
    ds.close()
    print(f"benchmark completed: {args.output}", flush=True)


if __name__ == "__main__":
    main()
