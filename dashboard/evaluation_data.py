"""Validated loading for saved synthetic-evaluation artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class EvaluationDataError(RuntimeError):
    """Raised when saved evaluation output is missing or unusable."""


REQUIRED_TOP_LEVEL = {
    "hidden_state",
    "calibration",
    "per_scenario",
    "alarm_comparison",
    "sensor_fault_scenario_only",
}
REQUIRED_HIDDEN_STATE = {
    "accuracy",
    "macro_f1",
    "multiclass_brier",
    "confusion_matrix",
    "confusion_matrix_labels",
}


def load_evaluation_metrics(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate the evaluation JSON used by Results."""
    source = Path(path)
    try:
        metrics = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EvaluationDataError(
            f"Evaluation metrics were not found at {source}. Run `python -m evaluation.evaluate`."
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationDataError(
            f"Evaluation metrics at {source} could not be read. Regenerate them with "
            "`python -m evaluation.evaluate`."
        ) from exc

    if not isinstance(metrics, dict):
        raise EvaluationDataError("Evaluation metrics must be a JSON object.")
    missing = REQUIRED_TOP_LEVEL - set(metrics)
    if missing:
        raise EvaluationDataError(
            "Evaluation metrics are incomplete; missing: " + ", ".join(sorted(missing))
        )
    hidden = metrics.get("hidden_state")
    if not isinstance(hidden, dict) or REQUIRED_HIDDEN_STATE - set(hidden):
        raise EvaluationDataError("Hidden-state evaluation metrics are incomplete.")
    return metrics
