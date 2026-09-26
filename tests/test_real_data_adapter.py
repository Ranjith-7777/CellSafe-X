"""Tests for the Phase-6 real-data adapter (data/real_data_adapter.py).

Tests that need the 215 MB source file are skipped when it has not been
downloaded (see data/external/warwick_thermal_runaway/download.py) - the
215 pre-existing tests, and the causal/normalisation/missing-evidence unit
tests below that use small synthetic fixtures, never require it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import model_parameters as P
from data.real_data_adapter import (
    N_EXPERIMENTS,
    TEMP_CHANNELS,
    VARIABLE_NAMES,
    _causal_bin_mean,
    build_observation_stream,
    event_markers,
    raw_data_available,
)
from models.bayesian_filter import BayesianFilter, log_normalise

requires_raw_data = pytest.mark.skipif(
    not raw_data_available(),
    reason="Warwick dataset not downloaded - see data/external/warwick_thermal_runaway/download.py",
)


# ---------------------------------------------------------------------------
# causal aggregation / time alignment (small synthetic fixtures - no real
# data required)
# ---------------------------------------------------------------------------
def test_causal_bin_mean_only_uses_samples_up_to_the_bin_edge():
    time_s = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    values = np.array([1.0, 2.0, 3.0, 4.0, 100.0])  # the 100.0 is in the future
    edges = np.array([0.0, 2.0, 4.0])
    means = _causal_bin_mean(time_s, values, edges)
    # bin (0,2] contains samples at 0.5, 1.5 -> mean 1.5
    assert means[0] == pytest.approx(1.5)
    # bin (2,4] contains samples at 2.5, 3.5 -> mean 3.5, NOT influenced by 4.5's 100.0
    assert means[1] == pytest.approx(3.5)


def test_causal_bin_mean_empty_bin_is_nan_not_zero():
    time_s = np.array([0.5, 10.5])
    values = np.array([5.0, 6.0])
    edges = np.array([0.0, 2.0, 4.0, 6.0])
    means = _causal_bin_mean(time_s, values, edges)
    assert means[0] == pytest.approx(5.0)
    assert np.isnan(means[1])  # no sample fell in (2,4]
    assert np.isnan(means[2])  # no sample fell in (4,6]


def test_temperature_rate_is_causal_first_difference():
    df = pd.DataFrame({
        "time_s": [10.0, 20.0, 30.0],
        "MidIntTemp": [20.0, 26.0, 20.0],  # rises then falls
        "CellVoltage": [4.1, 4.1, 4.1],
    })
    stream = build_observation_stream(df)
    assert stream[0]["temp_rate"] == pytest.approx(0.0)  # no prior sample
    dt_min = 10.0 / 60.0
    assert stream[1]["temp_rate"] == pytest.approx((26.0 - 20.0) / dt_min)
    assert stream[2]["temp_rate"] == pytest.approx((20.0 - 26.0) / dt_min)  # negative, not clipped


# ---------------------------------------------------------------------------
# missing-evidence handling (uses the EXISTING, unmodified filter interface)
# ---------------------------------------------------------------------------
def test_observation_stream_never_includes_unavailable_channels():
    df = pd.DataFrame({
        "time_s": [10.0, 20.0],
        "MidIntTemp": [20.0, 21.0],
        "CellVoltage": [4.1, 4.1],
    })
    stream = build_observation_stream(df)
    for obs in stream:
        assert "log_gas" not in obs
        assert "neighbour_c" not in obs
        assert "voltage_dev" not in obs  # excluded - see build_observation_stream docstring
        assert set(obs) <= set(P.OBS_CHANNELS)


def test_missing_channel_still_normalises_posterior():
    """A partial observation dict (only temp_c, temp_rate) must still update
    the filter to a valid, normalised posterior - this exercises the
    EXISTING observation_log_likelihood behaviour, unmodified by Phase 6."""
    f = BayesianFilter()
    step = f.update({"temp_c": 25.0, "temp_rate": 0.5})
    assert step.posterior.sum() == pytest.approx(1.0)
    assert (step.posterior >= 0).all()


def test_nan_values_in_a_partial_observation_are_skipped_like_missing():
    f = BayesianFilter()
    step = f.update({"temp_c": 25.0, "temp_rate": float("nan")})
    assert step.posterior.sum() == pytest.approx(1.0)
    assert np.isfinite(step.posterior).all()


def test_event_markers_are_never_treated_as_hidden_state_ground_truth():
    df = pd.DataFrame({
        "time_s": np.arange(10.0, 210.0, 10.0),
        "MidIntTemp": np.linspace(20.0, 200.0, 20),
        "IntPre": np.zeros(20),
    })
    ev = event_markers(df)
    assert ev.temp_runaway_onset_s is not None
    assert "not" in ev.note.lower() or "signal-derived" in ev.note.lower()


# ---------------------------------------------------------------------------
# raw parsing / full pipeline (requires the downloaded dataset)
# ---------------------------------------------------------------------------
@requires_raw_data
def test_raw_mat_parses_expected_variable_names():
    from data.real_data_adapter import _parse_raw

    data = _parse_raw()
    assert list(data.keys()) == VARIABLE_NAMES
    assert all(len(v) == N_EXPERIMENTS for v in data.values())


@requires_raw_data
def test_resampled_temperatures_are_physically_plausible():
    from data.real_data_adapter import resample_experiment

    for exp_id in range(N_EXPERIMENTS):
        df = resample_experiment(exp_id)
        vals = df["MidIntTemp"].dropna()
        assert len(vals) > 10
        # ambient start, thermal-runaway peak - real cylindrical-cell physics
        assert vals.min() > 0.0
        assert vals.max() < 1500.0
        assert (df["time_s"].diff().dropna() > 0).all()  # strictly increasing, causal


@requires_raw_data
def test_replay_is_deterministic():
    from evaluation.real_validation import replay_experiment

    a = replay_experiment(0)["trajectory"]
    b = replay_experiment(0)["trajectory"]
    pd.testing.assert_frame_equal(a, b)


@requires_raw_data
def test_replay_posteriors_all_normalise():
    from evaluation.real_validation import replay_experiment

    traj = replay_experiment(1)["trajectory"]
    total = traj[["p_healthy", "p_abnormal", "p_pre_runaway", "p_thermal_runaway"]].sum(axis=1)
    assert np.allclose(total, 1.0, atol=1e-9)


@requires_raw_data
def test_root_cause_and_intervention_are_not_run_on_real_data():
    """Guard against silently adding unsupported analyses later: real-data
    replay must not import or call root-cause / intervention / propagation /
    active-sensing machinery, since none of it is validated by this dataset."""
    import inspect

    import evaluation.real_validation as rv

    source = inspect.getsource(rv)
    for forbidden in ("diagnose_root_cause", "forecast_under_intervention", "forecast_pack_propagation"):
        assert forbidden not in source
