"""Run packaged-data experiments with: python main.py."""
import os
from pathlib import Path
import sys

# Must precede torch import for deterministic CUDA recurrent operations.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from clinical_multimodal.runner import main


if __name__ == "__main__":
    main()
