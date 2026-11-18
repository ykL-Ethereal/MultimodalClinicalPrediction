"""Create an immutable, label-blind, less densely sampled fixed tensor package."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from clinical_multimodal.data.resample import resample_cohort


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minutes", type=int, default=30)
    args = parser.parse_args()
    print(resample_cohort(args.source, args.output, args.minutes), flush=True)
