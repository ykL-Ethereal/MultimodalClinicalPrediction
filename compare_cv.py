"""Compare Train-only CV summaries in matched experiment groups; never open Test reports."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from clinical_multimodal.io import write_json
from clinical_multimodal.evaluation.cv_selection import compare_summaries


def compare(root):
    return compare_summaries(sorted(Path(root).rglob("cv_summary.json")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/patient_cv_30min"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new comparison output")
    groups = compare(args.runs_dir)
    if not groups:
        raise ValueError("No completed CV summaries found")
    write_json(args.output, groups)
    for key, group in groups.items():
        print(f"comparison_group={key[:12]}")
        for row in group["candidates"]:
            print(f"{row['model']:5} {row['sampling']:22} FPR={row['mean_fpr']:.6f} SD={row['std_fpr']:.6f} AP={row['mean_auprc']:.6f}")
