"""One invocation, predeclared model selection, retained CV and final artifacts."""
from copy import deepcopy
from pathlib import Path
import traceback

from ..evaluation.cv_selection import compare_summaries, RANKING_RULE
from ..io import fingerprint, read_json, write_json
from ..run_logging import create_run_dir, run_logging
from .patient_cv import cv_pipeline, finalize_cv


def experiment_pipeline(config, horizon_configs):
    config = deepcopy(config)
    config.validate()
    plans = [deepcopy(c) for c in horizon_configs]
    if config.training.protocol != "patient_cv" or not plans:
        raise ValueError("Automatic model selection requires patient_cv and at least one horizon")
    if len({c.data.horizon for c in plans}) != len(plans):
        raise ValueError("Duplicate horizons in experiment plan")
    for plan in plans:
        plan.validate()
        if plan.training.protocol != "patient_cv" or plan.data.horizons:
            raise ValueError("Experiment plans must be independent single-horizon CV configurations")
    output = create_run_dir(Path(config.training.output_dir) / "experiments", "patient_cv")
    state = {"status": "running", "stage": "planned", "ranking_rule": RANKING_RULE,
             "horizon_order": [c.data.horizon for c in plans], "horizons": [],
             "selection_uses_only_current_invocation_cv": True,
             "intermediate_results_retained": True}
    # Persist candidates/budgets and tie-breaking before fitting or reading any test result.
    write_json(output / "experiment_plan.json", {"config": config.to_dict(), "ranking_rule": RANKING_RULE,
                                                "plans": [c.to_dict() for c in plans]})
    write_json(output / "experiment_state.json", state)
    with run_logging(output, config):
        try:
            for plan in plans:
                hroot = output / f"h{plan.data.horizon}"
                hroot.mkdir()
                plan.training.output_dir = str(hroot)
                write_json(hroot / "config.json", plan.to_dict())
                current = {"horizon_hours": plan.data.horizon, "status": "running", "stage": "cross_validation",
                           "planned_models": plan.training.models.copy(), "cv_summaries": []}
                state["horizons"].append(current)
                state["stage"] = "cross_validation"
                write_json(output / "experiment_state.json", state)
                for name in plan.training.models:
                    current["active_model"] = name
                    write_json(output / "experiment_state.json", state)
                    candidate = deepcopy(plan)
                    candidate.training.models = [name]
                    candidate.model.vital_encoder = name
                    summary_path = cv_pipeline(candidate)
                    current["cv_summaries"].append(str(summary_path.resolve()))
                    write_json(output / "experiment_state.json", state)
                groups = compare_summaries(current["cv_summaries"])
                if len(groups) != 1:
                    raise ValueError("Candidate data/code/folds/budgets differ; refusing automatic finalization")
                ranking = next(iter(groups.values()))
                if sorted(r["model"] for r in ranking["candidates"]) != sorted(plan.training.models):
                    raise ValueError("CV candidates do not match the predeclared model list")
                selection = {**ranking, "horizon_hours": plan.data.horizon,
                             "winner": ranking["candidates"][0], "selection_source": "current_invocation_train_cv",
                             "validation_used": False, "test_used": False}
                selection["sha256"] = fingerprint(selection)
                write_json(hroot / "model_selection.json", selection)
                current.update(stage="finalization", selected_model=selection["winner"]["model"],
                               selection_record=str((hroot / "model_selection.json").resolve()))
                state["stage"] = "finalization"
                write_json(output / "experiment_state.json", state)
                print(f"horizon={plan.data.horizon}h selected={current['selected_model']} "
                      f"CV_mean_FPR={selection['winner']['mean_fpr']:.6f}", flush=True)
                artifact = finalize_cv(selection["winner"]["summary"], hroot)
                current.update(status="completed", stage="completed", artifact=str(artifact.resolve()),
                               final_summary=str((artifact.parent / "run_summary.json").resolve()))
                current.pop("active_model", None)
                write_json(output / "experiment_state.json", state)
            state.update(status="completed", stage="completed")
            write_json(output / "experiment_state.json", state)
        except BaseException as exc:
            state.update(status="failed", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
            if state["horizons"] and state["horizons"][-1]["status"] != "completed":
                state["horizons"][-1]["status"] = "failed"
            write_json(output / "experiment_state.json", state)
            raise
    return output / "experiment_state.json"
