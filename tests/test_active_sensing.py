"""Tests for Phase-4 active sensing / Value of Information
(models/active_sensing.py)."""

from __future__ import annotations

import numpy as np
import pytest

from config import model_parameters as P
from models.active_sensing import (
    expected_information_gain,
    expected_value_of_information,
    posterior_entropy,
    rank_information_gain,
    recommend_measurement,
)

AMBIGUOUS = [0.30, 0.40, 0.20, 0.10]
CONFIDENT = [0.001, 0.001, 0.003, 0.995]
UNIFORM = [0.25, 0.25, 0.25, 0.25]


# ---------------------------------------------------------------------------
# entropy
# ---------------------------------------------------------------------------
def test_entropy_of_one_hot_belief_is_zero():
    assert posterior_entropy([1.0, 0.0, 0.0, 0.0]) == pytest.approx(0.0, abs=1e-9)


def test_entropy_of_uniform_belief_is_ln_n_states():
    assert posterior_entropy(UNIFORM) == pytest.approx(np.log(P.N_STATES), rel=1e-6)


def test_entropy_is_nonnegative():
    for b in (AMBIGUOUS, CONFIDENT, UNIFORM):
        assert posterior_entropy(b) >= 0.0


# ---------------------------------------------------------------------------
# expected information gain
# ---------------------------------------------------------------------------
def test_unknown_channel_rejected():
    with pytest.raises(ValueError):
        expected_information_gain(AMBIGUOUS, "current")  # not in the emission model


def test_eig_is_nonnegative_within_monte_carlo_tolerance():
    for channel in P.ACTIVE_SENSING_CANDIDATE_CHANNELS:
        r = expected_information_gain(AMBIGUOUS, channel, n_samples=2000)
        assert r.expected_information_gain >= -0.01  # MC tolerance


def test_eig_deterministic_with_fixed_seed():
    a = expected_information_gain(AMBIGUOUS, "temp_c", n_samples=1000, seed=123)
    b = expected_information_gain(AMBIGUOUS, "temp_c", n_samples=1000, seed=123)
    assert a.expected_information_gain == b.expected_information_gain


def test_different_seeds_give_close_but_not_identical_estimates():
    a = expected_information_gain(AMBIGUOUS, "temp_c", n_samples=1000, seed=1)
    b = expected_information_gain(AMBIGUOUS, "temp_c", n_samples=1000, seed=2)
    assert a.expected_information_gain == pytest.approx(b.expected_information_gain, abs=0.15)


def test_uninformative_synthetic_channel_has_near_zero_eig(monkeypatch):
    """A channel with an IDENTICAL Gaussian across every state carries zero
    discriminative power by construction - a direct check that EIG reflects
    the emission model rather than an arbitrary weight."""
    dummy = "dummy_uninformative"
    monkeypatch.setattr(P, "ACTIVE_SENSING_CANDIDATE_CHANNELS", P.OBS_CHANNELS + (dummy,))
    monkeypatch.setitem(P.OBS_LIKELIHOOD_WEIGHTS, dummy, 0.5)
    for state in P.STATES:
        monkeypatch.setitem(P.OBS_MEANS[state], dummy, 50.0)
        monkeypatch.setitem(P.OBS_STDS[state], dummy, 10.0)
    r = expected_information_gain(AMBIGUOUS, dummy, n_samples=3000)
    assert abs(r.expected_information_gain) < 0.02


def test_informative_channel_ranks_above_uninformative_control(monkeypatch):
    dummy = "dummy_uninformative"
    monkeypatch.setattr(P, "ACTIVE_SENSING_CANDIDATE_CHANNELS", P.OBS_CHANNELS + (dummy,))
    monkeypatch.setitem(P.OBS_LIKELIHOOD_WEIGHTS, dummy, 0.5)
    for state in P.STATES:
        monkeypatch.setitem(P.OBS_MEANS[state], dummy, 50.0)
        monkeypatch.setitem(P.OBS_STDS[state], dummy, 10.0)
    informative = expected_information_gain(AMBIGUOUS, "temp_c", n_samples=3000)
    uninformative = expected_information_gain(AMBIGUOUS, dummy, n_samples=3000)
    assert informative.expected_information_gain > uninformative.expected_information_gain


def test_ambiguous_posterior_has_more_eig_headroom_than_confident_posterior():
    """EIG is capped above by current entropy, so a near-certain posterior
    cannot offer as much expected information gain as an ambiguous one for
    the same, informative channel."""
    amb = expected_information_gain(AMBIGUOUS, "temp_c", n_samples=3000)
    conf = expected_information_gain(CONFIDENT, "temp_c", n_samples=3000)
    assert amb.expected_information_gain > conf.expected_information_gain


def test_rank_information_gain_is_sorted_descending():
    ranking = rank_information_gain(AMBIGUOUS)
    values = [r.expected_information_gain for r in ranking]
    assert values == sorted(values, reverse=True)
    assert {r.channel for r in ranking} == set(P.ACTIVE_SENSING_CANDIDATE_CHANNELS)


# ---------------------------------------------------------------------------
# expected value of information
# ---------------------------------------------------------------------------
def test_evi_is_nonnegative_within_monte_carlo_tolerance():
    r = expected_value_of_information(AMBIGUOUS, "temp_c", n_samples=2000)
    assert r.expected_value_of_information >= -0.5  # loss units, generous MC tolerance


def test_evi_and_eig_rankings_are_not_forced_to_agree():
    """EIG and EVI answer different questions and must be computed
    independently - this just checks both are produced without one being
    silently derived from the other."""
    eig = expected_information_gain(AMBIGUOUS, "temp_c", n_samples=1000)
    evi = expected_value_of_information(AMBIGUOUS, "temp_c", n_samples=1000)
    assert eig.expected_information_gain != evi.expected_value_of_information


# ---------------------------------------------------------------------------
# gated recommendation / Backup Measurement integration
# ---------------------------------------------------------------------------
def test_confident_posterior_does_not_recommend_measurement():
    rec = recommend_measurement(CONFIDENT)
    assert rec.should_request_measurement is False


def test_ambiguous_posterior_recommends_measurement():
    rec = recommend_measurement(AMBIGUOUS)
    assert rec.should_request_measurement is True


def test_emergency_dangerous_posterior_does_not_recommend_measurement():
    dangerous_but_ambiguous = [0.0, 0.02, 0.08, 0.90]
    rec = recommend_measurement(dangerous_but_ambiguous)
    assert rec.should_request_measurement is False
    assert any("unsafe" in r or "emergency" in r.lower() for r in [x.lower() for x in rec.reasons])


def test_recommendation_reports_best_channel_from_ranking():
    rec = recommend_measurement(AMBIGUOUS)
    assert rec.best_channel == rec.information_gain[0].channel


def test_request_backup_measurement_still_has_zero_physical_effect():
    """Phase 3 invariant, re-checked here: this module can recommend WHEN to
    measure, but the intervention forecast for Request Backup Measurement
    must remain physically neutral - the two capabilities are separate."""
    from models.intervention import forecast_under_intervention

    for belief in (AMBIGUOUS, CONFIDENT, UNIFORM):
        cf = forecast_under_intervention(belief, "Request Backup Measurement", horizon_min=15.0)
        assert cf.risk_reduction == pytest.approx(0.0, abs=1e-9)
