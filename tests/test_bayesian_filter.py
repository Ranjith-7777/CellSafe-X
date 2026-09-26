"""Tests for the exact Bayesian filter and the multi-step risk forecast."""

from __future__ import annotations

import numpy as np
import pytest

from config import model_parameters as P
from models.bayesian_filter import (
    BayesianFilter,
    calibrated_posterior,
    log_normalise,
    log_sum_exp,
    observation_log_likelihood,
)
from models.risk_forecast import (
    dangerous_probability,
    forecast_risk,
    pack_risk,
    propagate,
    risk_trajectory,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def obs_for(state: str, **overrides) -> dict:
    """A textbook observation sitting exactly on a state's mean."""
    o = dict(P.OBS_MEANS[state])
    o.update(overrides)
    return o


HEALTHY_OBS = obs_for("Healthy")
RUNAWAY_OBS = obs_for("Thermal Runaway")


# ---------------------------------------------------------------------------
# model sanity
# ---------------------------------------------------------------------------
def test_transition_matrix_rows_sum_to_one():
    assert np.allclose(P.TRANSITION_MATRIX.sum(axis=1), 1.0)
    assert (P.TRANSITION_MATRIX >= 0).all()


def test_initial_distribution_sums_to_one():
    assert P.INITIAL_STATE_PROBS.sum() == pytest.approx(1.0)


def test_runaway_is_nearly_absorbing():
    assert P.TRANSITION_MATRIX[P.S_RUNAWAY, P.S_RUNAWAY] >= 0.99


def test_healthy_cannot_jump_straight_to_dangerous_states():
    assert P.TRANSITION_MATRIX[P.S_HEALTHY, P.S_PRE] == 0.0
    assert P.TRANSITION_MATRIX[P.S_HEALTHY, P.S_RUNAWAY] == 0.0


def test_recovery_from_abnormal_heating_is_possible():
    assert P.TRANSITION_MATRIX[P.S_ABNORMAL, P.S_HEALTHY] > 0.0


# ---------------------------------------------------------------------------
# posterior normalisation
# ---------------------------------------------------------------------------
def test_posterior_sums_to_one_every_step():
    f = BayesianFilter()
    rng = np.random.default_rng(0)
    for _ in range(200):
        state = P.STATES[rng.integers(0, P.N_STATES)]
        obs = {
            ch: P.OBS_MEANS[state][ch] + rng.normal(0, P.OBS_STDS[state][ch])
            for ch in P.OBS_CHANNELS
        }
        step = f.update(obs)
        assert step.posterior.sum() == pytest.approx(1.0, abs=1e-12)
        assert step.prior.sum() == pytest.approx(1.0, abs=1e-12)
        assert (step.posterior >= 0).all()


def test_log_normalise_returns_a_distribution():
    v = log_normalise(np.array([-1000.0, -1001.0, -1002.0, -1003.0]))
    assert v.sum() == pytest.approx(1.0)
    assert (v > 0).all()


def test_log_normalise_survives_total_underflow():
    """Every log-weight at -inf must give a uniform fallback, not NaN."""
    v = log_normalise(np.full(4, -np.inf))
    assert v.sum() == pytest.approx(1.0)
    assert np.allclose(v, 0.25)


def test_log_sum_exp_matches_naive_computation():
    v = np.array([-2.0, -3.5, -1.25, -7.0])
    assert log_sum_exp(v) == pytest.approx(float(np.log(np.exp(v).sum())))


# ---------------------------------------------------------------------------
# numerical stability
# ---------------------------------------------------------------------------
def test_filter_is_stable_over_a_long_run_of_extreme_evidence():
    """5000 steps of an observation far outside every state's support.

    Raw (non-log) filtering would underflow to all-zeros here and produce NaNs.
    """
    f = BayesianFilter()
    extreme = {ch: 1e4 for ch in P.OBS_CHANNELS}
    for _ in range(5000):
        step = f.update(extreme)
        assert np.isfinite(step.posterior).all()
        assert step.posterior.sum() == pytest.approx(1.0, abs=1e-10)


def test_filter_handles_missing_channels():
    """A missing observation must contribute no evidence, not crash."""
    f = BayesianFilter()
    full = f.update(HEALTHY_OBS).posterior.copy()

    g = BayesianFilter()
    partial = g.update({"temp_c": HEALTHY_OBS["temp_c"]}).posterior

    assert partial.sum() == pytest.approx(1.0)
    # Fewer channels means weaker evidence, so the partial posterior must stay
    # closer to the prior than the full one does.
    prior = P.INITIAL_STATE_PROBS
    assert np.abs(partial - prior).sum() < np.abs(full - prior).sum()


def test_nonfinite_observation_is_ignored():
    ll_with_nan = observation_log_likelihood({**HEALTHY_OBS, "temp_c": float("nan")})
    obs_without = {k: v for k, v in HEALTHY_OBS.items() if k != "temp_c"}
    assert np.allclose(ll_with_nan, observation_log_likelihood(obs_without))


# ---------------------------------------------------------------------------
# evidence drives the posterior in the right direction
# ---------------------------------------------------------------------------
def test_consistent_danger_evidence_raises_dangerous_probability():
    f = BayesianFilter()
    start = dangerous_probability(f.posterior)

    danger_curve = []
    for _ in range(40):
        step = f.update(RUNAWAY_OBS)
        danger_curve.append(dangerous_probability(step.posterior))

    assert danger_curve[-1] > start
    assert danger_curve[-1] > 0.9
    # The rise must be monotone: sustained consistent evidence should never
    # make the filter less worried.
    assert all(b >= a - 1e-9 for a, b in zip(danger_curve, danger_curve[1:]))


def test_healthy_evidence_keeps_dangerous_probability_low():
    f = BayesianFilter()
    for _ in range(60):
        step = f.update(HEALTHY_OBS)
    assert dangerous_probability(step.posterior) < 0.01
    assert int(np.argmax(step.posterior)) == P.S_HEALTHY


def test_filter_recovers_when_evidence_returns_to_normal():
    """Limited recovery from Abnormal Heating must actually be reachable."""
    f = BayesianFilter()
    for _ in range(30):
        f.update(obs_for("Abnormal Heating"))
    assert int(np.argmax(f.posterior)) == P.S_ABNORMAL

    for _ in range(60):
        f.update(HEALTHY_OBS)
    assert int(np.argmax(f.posterior)) == P.S_HEALTHY


def test_reset_restores_the_prior():
    f = BayesianFilter()
    for _ in range(20):
        f.update(RUNAWAY_OBS)
    f.reset()
    assert np.allclose(f.posterior, P.INITIAL_STATE_PROBS)
    assert f.n_updates == 0


def test_rejects_an_invalid_transition_matrix():
    bad = np.full((P.N_STATES, P.N_STATES), 0.5)
    with pytest.raises(ValueError):
        BayesianFilter(transition=bad)


# ---------------------------------------------------------------------------
# forecasting
# ---------------------------------------------------------------------------
def test_forecast_probabilities_are_valid():
    for belief in (
        P.INITIAL_STATE_PROBS,
        np.array([0.0, 0.0, 0.0, 1.0]),
        np.array([0.25, 0.25, 0.25, 0.25]),
        np.array([0.1, 0.6, 0.25, 0.05]),
    ):
        fc = forecast_risk(belief)
        assert 0.0 <= fc.current_dangerous <= 1.0
        for h in P.FORECAST_HORIZONS_MIN:
            p = fc.dangerous_by_horizon[float(h)]
            assert 0.0 <= p <= 1.0
            dist = fc.distribution_by_horizon[float(h)]
            assert dist.sum() == pytest.approx(1.0)
            assert (dist >= 0).all()


def test_propagation_matches_repeated_multiplication():
    """The cached matrix power must equal the naive step-by-step loop."""
    b = np.array([0.4, 0.3, 0.2, 0.1])
    naive = b.copy()
    for _ in range(37):
        naive = naive @ P.TRANSITION_MATRIX
        naive /= naive.sum()
    assert np.allclose(propagate(b, 37), naive, atol=1e-12)


def test_forecast_from_runaway_stays_high():
    fc = forecast_risk(np.array([0.0, 0.0, 0.0, 1.0]))
    for h in P.FORECAST_HORIZONS_MIN:
        assert fc.dangerous_by_horizon[float(h)] > 0.9


def test_forecast_from_healthy_stays_low():
    """A healthy pack must not forecast an alarming 30-minute risk from the
    prior alone - that would make the forecast panel meaningless."""
    fc = forecast_risk(np.array([1.0, 0.0, 0.0, 0.0]))
    assert fc.dangerous_by_horizon[30.0] < 0.25


def test_longer_horizons_are_more_dangerous_from_a_healthy_start():
    fc = forecast_risk(np.array([1.0, 0.0, 0.0, 0.0]))
    values = [fc.dangerous_by_horizon[float(h)] for h in P.FORECAST_HORIZONS_MIN]
    assert values == sorted(values)


def test_zero_step_propagation_is_the_identity():
    b = np.array([0.4, 0.3, 0.2, 0.1])
    assert np.allclose(propagate(b, 0), b)


def test_propagate_rejects_a_zero_belief():
    with pytest.raises(ValueError):
        propagate(np.zeros(P.N_STATES), 5)


def test_risk_trajectory_is_bounded_and_correctly_shaped():
    traj = risk_trajectory(np.array([0.2, 0.5, 0.2, 0.1]), max_minutes=30.0, n_points=31)
    assert len(traj["minutes"]) == 31
    assert len(traj["dangerous_probability"]) == 31
    assert ((traj["dangerous_probability"] >= 0) & (traj["dangerous_probability"] <= 1)).all()
    assert traj["dangerous_probability"][0] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# posterior calibration (temperature scaling)
# ---------------------------------------------------------------------------
def test_calibrated_posterior_identity_at_temperature_one():
    p = np.array([0.7, 0.2, 0.08, 0.02])
    assert np.allclose(calibrated_posterior(p, 1.0), p)


def test_calibrated_posterior_sums_to_one():
    for t in (0.3, 0.5, 1.0, 2.0, 5.0):
        p = np.array([0.9997, 0.0002, 0.0001, 0.0])
        cal = calibrated_posterior(p, t)
        assert cal.sum() == pytest.approx(1.0)
        assert (cal >= 0).all()


def test_calibrated_posterior_preserves_argmax():
    """Temperature scaling must never change the MAP state - only accuracy-
    independent metrics (Brier, NLL, ECE) may move."""
    p = np.array([0.05, 0.15, 0.55, 0.25])
    for t in (0.2, 0.5, 1.0, 2.0, 4.0, 10.0):
        assert np.argmax(calibrated_posterior(p, t)) == np.argmax(p)


def test_calibrated_posterior_above_one_flattens_confidence():
    """temperature > 1 must reduce the max-probability entry (soften an
    over-confident posterior)."""
    p = np.array([0.999, 0.0007, 0.0002, 0.0001])
    cal = calibrated_posterior(p, 3.0)
    assert cal.max() < p.max()


def test_calibrated_posterior_below_one_sharpens_confidence():
    p = np.array([0.6, 0.25, 0.1, 0.05])
    cal = calibrated_posterior(p, 0.4)
    assert cal.max() > p.max()


def test_calibrated_posterior_rejects_nonpositive_temperature():
    with pytest.raises(ValueError):
        calibrated_posterior(np.array([0.25, 0.25, 0.25, 0.25]), 0.0)


def test_pack_risk_aggregation():
    safe = np.array([1.0, 0.0, 0.0, 0.0])
    hot = np.array([0.0, 0.0, 0.3, 0.7])
    pr = pack_risk([safe, safe, hot, safe, safe, safe])
    assert pr["worst_cell_index"] == 2
    assert pr["worst_cell_dangerous"] == pytest.approx(1.0)
    assert 0.0 <= pr["any_cell_dangerous"] <= 1.0
    assert pr["any_cell_dangerous"] >= pr["worst_cell_dangerous"] - 1e-12
