from __future__ import annotations

from pathlib import Path
import os
import tempfile
import time
import numpy as np

from ..io import fingerprint
from ..profiling import memory_snapshot
from .text import clean_text


class TextCache:
    def __init__(self, root, backend):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        self.hits = self.misses = 0
        self.hit_seconds = self.miss_seconds = 0.0
        self.hit_memory = self.miss_memory = None

    def key(self, text):
        return fingerprint({"text": clean_text(text), "encoder": self.backend.identity})

    def get(self, text):
        start = time.perf_counter()
        path = self.root / (self.key(text) + ".npy")
        if path.exists():
            result = np.load(path, allow_pickle=False)
            self._validate(result)
            self.hits += 1
            self.hit_seconds += time.perf_counter() - start
            self.hit_memory = self._memory_snapshot()
            return result
        result = self.backend.encode(clean_text(text)).astype(np.float32)
        self._validate(result)
        with tempfile.NamedTemporaryFile(dir=self.root, suffix=".tmp", delete=False) as f:
            temporary = Path(f.name)
            np.save(f, result, allow_pickle=False)
        try:
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        self.misses += 1
        self.miss_seconds += time.perf_counter() - start
        self.miss_memory = self._memory_snapshot()
        return result

    def _validate(self, value):
        if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] != self.backend.dimension or not np.isfinite(value).all():
            raise ValueError("Corrupt/incompatible text cache entry")

    def _memory_snapshot(self):
        result = {**memory_snapshot(), "cuda_allocated_bytes": None,
                  "cuda_peak_allocated_bytes_since_reset": None}
        device = getattr(self.backend, "device", None)
        if device is not None and device.type == "cuda":
            cuda = self.backend.torch.cuda
            result.update(cuda_allocated_bytes=int(cuda.memory_allocated(device)),
                          cuda_peak_allocated_bytes_since_reset=int(cuda.max_memory_allocated(device)))
        return result

    def stats(self):
        return {"hits": self.hits, "misses": self.misses, "hit_seconds": self.hit_seconds,
                "miss_seconds": self.miss_seconds, "last_hit_process_memory": self.hit_memory,
                "last_miss_process_memory": self.miss_memory}
