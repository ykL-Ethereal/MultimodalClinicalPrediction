"""Single command entry point for the packaged fixed cohort."""
import argparse
from copy import deepcopy
from pathlib import Path

from .config import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train TCN/LSTM/GRU/GRU-D on the packaged fixed cohort")
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).resolve().parents[2] / "configs" / "train.yaml")
    parser.add_argument("--models", nargs="+", choices=["tcn", "lstm", "gru", "grud"],
                        help="Override training.models in the configuration")
    parser.add_argument("--validate-only", action="store_true", help="Validate the fixed package without training")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.models:
        config.training.models = args.models
        config.model.vital_encoder = args.models[0]
    config.validate()
    if args.validate_only:
        from .pipelines.prepare import prepare
        from .run_logging import create_run_dir, run_logging
        output = create_run_dir(config.training.output_dir, "validation")
        config.training.output_dir = str(output)
        with run_logging(output, config):
            bundle, _, _, _ = prepare(config, encode_text=False)
            print(f"Validated {len(bundle.metadata)} samples in {len(bundle.starts)} shards.")
        return
    from .pipelines.train import train_pipeline
    for name in config.training.models:
        selected = deepcopy(config)
        selected.model.vital_encoder = name
        print(train_pipeline(selected), flush=True)
