from ..io import write_json


def save_report(path, metrics, *, split, extra=None):
    report = {"purpose": "research_evaluation",
              "split": split, "metrics": metrics, **(extra or {})}
    write_json(path, report)
    return report
