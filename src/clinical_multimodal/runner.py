"""Single command entry point for the packaged fixed cohort."""
import argparse
from copy import deepcopy
from pathlib import Path

from .config import load_config


def horizon_configs(config):
    """Expand the ordered schedule into isolated single-horizon run configurations."""
    for horizon in config.data.horizons or [config.data.horizon]:
        selected = deepcopy(config)
        selected.data.horizon = horizon
        selected.data.horizons = []
        selected.training.positive_fraction = config.training.positive_fraction_by_horizon.get(
            horizon, config.training.positive_fraction)
        selected.training.positive_fraction_by_horizon = {}
        yield selected


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run TCN/LSTM/GRU/GRU-D/NEWS2/XGBoost on the packaged fixed cohort")
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).resolve().parents[2] / "configs" / "patient_cv_30min.yaml")
    parser.add_argument("--models", nargs="+", choices=["tcn", "lstm", "gru", "grud", "news2", "xgboost"],
                        help="Override training.models in the configuration")
    parser.add_argument("--vital_only", "--vital-only", action="store_true",
                        help="Use only the 20 vital channels; exclude labs, notes and static features")
    parser.add_argument("--validate-only", action="store_true", help="Validate the fixed package without training")
    parser.add_argument("--audit-threshold-splits", action="store_true", help="Audit validation patient partitions without training")
    parser.add_argument("--horizon", type=int, choices=[1, 2, 3, 6], help="One horizon, one independently selected threshold")
    parser.add_argument("--output-dir", type=Path, help="New run root (relative to current working directory)")
    parser.add_argument("--artifact", type=Path, help="Evaluate saved decisions; never reselect thresholds")
    parser.add_argument("--evaluation-output", type=Path, help="New evaluation JSON path; existing files are rejected")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], help="Override training/evaluation device")
    parser.add_argument("--finalize-cv", type=Path, help="Freeze one selected CV summary, refit Train, threshold on Validation, evaluate Test")
    parser.add_argument("--cv-only", action="store_true", help="Only run patient CV; skip automatic model selection/finalization")
    args = parser.parse_args(argv)
    if args.finalize_cv:
        if any((args.models, args.horizon, args.device, args.artifact, args.validate_only, args.audit_threshold_splits, args.vital_only, args.cv_only)):
            parser.error("Finalization uses the frozen CV configuration; only --output-dir may be overridden")
        from .pipelines.patient_cv import finalize_cv
        print(finalize_cv(args.finalize_cv, args.output_dir), flush=True)
        return
    config = load_config(args.config, vital_only=True if args.vital_only else None, models=args.models)
    if args.horizon:
        config.data.horizon = args.horizon
        config.data.horizons = []  # An explicit CLI horizon overrides the entire schedule.
    if args.output_dir:
        config.training.output_dir = str(args.output_dir.resolve())
    if args.device:
        config.training.device = args.device
    if args.artifact:
        if not args.evaluation_output:
            parser.error("--artifact requires --evaluation-output")
        if args.evaluation_output.exists():
            raise FileExistsError("Choose a new evaluation output path")
        from .pipelines.evaluate import evaluate_pipeline
        print(evaluate_pipeline(args.artifact, config.data.root, args.evaluation_output,
                                device=args.device or "auto", batch_size=config.training.batch_size))
        return
    if args.evaluation_output:
        parser.error("--evaluation-output requires --artifact")
    if args.vital_only and not args.models:
        config.training.models = ["tcn", "lstm", "gru", "grud"]
        config.model.vital_encoder = "tcn"
    config.validate()
    if args.cv_only and config.training.protocol != "patient_cv":
        parser.error("--cv-only requires the patient_cv protocol")
    if args.audit_threshold_splits:
        from .pipelines.threshold import audit_threshold_splits
        for selected in horizon_configs(config):
            print(audit_threshold_splits(selected))
        return
    if args.validate_only:
        from .pipelines.prepare import prepare
        from .run_logging import create_run_dir, run_logging
        root = Path(config.training.output_dir) / "vital_only" if config.model.vital_only else config.training.output_dir
        output = create_run_dir(root, "validation")
        config.training.output_dir = str(output)
        with run_logging(output, config):
            bundle, _, _, _ = prepare(config, encode_text=False)
            print(f"Validated {len(bundle.metadata)} samples in {len(bundle.starts)} shards.")
        return
    if config.training.protocol == "patient_cv" and not args.cv_only:
        from .pipelines.experiment import experiment_pipeline
        print(experiment_pipeline(config, list(horizon_configs(config))), flush=True)
        return
    for horizon_config in horizon_configs(config):
        for name in horizon_config.training.models:
            selected = deepcopy(horizon_config)
            selected.training.models = [name]
            selected.model.vital_encoder = name
            if name == "news2":
                selected.model.vital_only = False  # NEWS2 retains its independent rules pipeline.
                from .pipelines.news2 import news2_pipeline
                print(news2_pipeline(selected), flush=True)
            else:
                from .pipelines.train import train_pipeline
                print(train_pipeline(selected), flush=True)
