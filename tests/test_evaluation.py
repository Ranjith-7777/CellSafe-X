"""Tests for the Phase-1 evaluation additions: detection-timing metrics,
calibration-temperature fitting/leakage separation, and the invariant that
nothing in the inference path depends on the simulator's scenario label."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

import models
import models.bayesian_filter
import models.decision_engine
import models.fault_diagnosis
from config import model_parameters as P
from evaluation.evaluate import (
    BASE_SEED,
    CALIBRATION_SEED_BASE,
    apply_temperature,
    detection_timing_metrics,
    fit_calibration_temperature,
)


# ---------------------------------------------------------------------------
# detection-timing metrics
# ---------------------------------------------------------------------------
def _timing_frame(rows):
    """Build a minimal single-run frame for detection_timing_metrics."""
    return pd.DataFrame(
        [
            {
                "scenario": "Synthetic",
                "seed": 1,
                "cell": 0,
                "step": i,
                "time_min": i * P.DT_MINUTES,
                "true_dangerous": td,
                "p_dangerous": pd_,
            }
            for i, (td, pd_) in enumerate(rows)
        ]
    )


def test_detection_timing_measures_positive_lag():
    # true state goes dangerous at step 2 (t=0.33min); alarm fires at step 5.
    rows = [(False, 0.0), (False, 0.05), (True, 0.1), (True, 0.2), (True, 0.3), (True, 0.6)]
    df = _timing_frame(rows)
    out = detection_timing_metrics(df)["per_scenario"]["Synthetic"]
    assert out["n_missed"] == 0
    expected_delay = (5 - 2) * P.DT_MINUTES
    assert out["mean_detection_delay_min"] == pytest.approx(expected_delay)


def test_detection_timing_negative_lag_is_early_warning():
    # alarm fires (step 1) BEFORE the ground truth actually crosses (step 3).
    rows = [(False, 0.0), (False, 0.6), (False, 0.6), (True, 0.6), (True, 0.6)]
    df = _timing_frame(rows)
    out = detection_timing_metrics(df)["per_scenario"]["Synthetic"]
    assert out["mean_detection_delay_min"] < 0


def test_detection_timing_flags_a_miss():
    rows = [(False, 0.0), (True, 0.1), (True, 0.2)]  # alarm never fires
    df = _timing_frame(rows)
    out = detection_timing_metrics(df)["per_scenario"]["Synthetic"]
    assert out["n_missed"] == 1
    assert out["miss_rate"] == pytest.approx(1.0)
    assert np.isnan(out["mean_detection_delay_min"])


def test_detection_timing_skips_runs_that_never_go_dangerous():
    rows = [(False, 0.0), (False, 0.1), (False, 0.05)]
    df = _timing_frame(rows)
    out = detection_timing_metrics(df)["per_scenario"]
    assert "Synthetic" not in out


# ---------------------------------------------------------------------------
# calibration temperature: fitting mechanics + leakage separation
# ---------------------------------------------------------------------------
def test_apply_temperature_is_identity_at_one():
    probs = np.array([[0.7, 0.2, 0.08, 0.02], [0.25, 0.25, 0.25, 0.25]])
    out = apply_temperature(probs, 1.0)
    assert np.allclose(out.sum(axis=1), 1.0)
    assert np.allclose(out, probs / probs.sum(axis=1, keepdims=True))


def test_apply_temperature_preserves_argmax_rowwise():
    probs = np.array([[0.05, 0.15, 0.55, 0.25], [0.9, 0.05, 0.03, 0.02]])
    for t in (0.3, 1.0, 3.0):
        out = apply_temperature(probs, t)
        assert (out.argmax(axis=1) == probs.argmax(axis=1)).all()


def test_calibration_fit_uses_a_seed_range_disjoint_from_evaluation_seeds():
    """The whole point of CALIBRATION_SEED_BASE is that it must never collide
    with the seeds the final metrics are reported on."""
    assert CALIBRATION_SEED_BASE != BASE_SEED
    n_eval_seeds_reasonable_upper_bound = 1000
    assert CALIBRATION_SEED_BASE >= BASE_SEED + n_eval_seeds_reasonable_upper_bound


def test_fit_calibration_temperature_returns_a_positive_value():
    result = fit_calibration_temperature(n_seeds=1, n_steps=30, verbose=False)
    assert result["temperature"] > 0
    assert result["fit_seed_base"] == CALIBRATION_SEED_BASE
    # fitting must not do worse (on its own set) than leaving it uncalibrated
    assert result["fit_nll_at_temperature"] <= result["fit_nll_uncalibrated"] + 1e-9


# ---------------------------------------------------------------------------
# no scenario-label leakage into inference
# ---------------------------------------------------------------------------
def test_pipeline_step_has_no_scenario_parameter():
    """The inference engine must not be able to see the simulator's scenario
    label - only raw sensor readings. Guards against a future change quietly
    special-casing a scenario name inside the pipeline."""
    for fn in (
        models.CellPipeline.step,
        models.fault_diagnosis.sensor_reliability,
        models.fault_diagnosis.diagnose_root_cause,
        models.decision_engine.recommend_action,
        models.bayesian_filter.observation_log_likelihood,
    ):
        params = set(inspect.signature(fn).parameters)
        assert not any("scenario" in p.lower() for p in params), fn

    import ast
    import pathlib

    models_dir = pathlib.Path(models.__file__).resolve().parent
    for py_file in models_dir.glob("*.py"):
        source = py_file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(py_file))
        names = {
            node.id.lower()
            for node in ast.walk(tree)
            if isinstance(node, ast.Name)
        }
        attrs = {
            node.attr.lower()
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
        }
        assert not any("scenario" in n for n in names | attrs), py_file
