"""Model-specific run directories and persistent stdout/stderr logs."""
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import traceback

from .io import write_json


def create_run_dir(root, model):
    timestamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    output = Path(root) / model / timestamp
    output.mkdir(parents=True, exist_ok=False)
    return output


class Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, text):
        self.console.write(text)
        self.log.write(text)
        self.log.flush()
        return len(text)

    def flush(self):
        self.console.flush()
        self.log.flush()


@contextmanager
def run_logging(output, config):
    write_json(output / "config.json", config.to_dict())
    with (output / "training.log").open("w", encoding="utf-8", buffering=1) as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            write_json(output / "status.json", {"status": "running", "model": config.model.vital_encoder})
            print(f"model={config.model.vital_encoder} horizon={config.data.horizon}h output={output}", flush=True)
            try:
                yield
            except BaseException as exc:
                traceback.print_exc()
                write_json(output / "status.json", {"status": "failed", "model": config.model.vital_encoder,
                                                    "error_type": type(exc).__name__, "error": str(exc)})
                raise
            else:
                write_json(output / "status.json", {"status": "completed", "model": config.model.vital_encoder})
