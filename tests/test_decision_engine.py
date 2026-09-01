"""Tests for the minimum-expected-loss decision engine and the threshold baseline."""

from __future__ import annotations

import numpy as np
import pytest

from config import model_parameters as P
from data.battery_simulator import simulate_pack
from models import CellPipeline, run_pipeline
from models.decision_engine import (
    compare_with_threshold,
    expected_losses,
    recommend_action,
    threshold_baseline,
)

HEALTHY = np.array([1.0, 0.0, 0.0, 0.0])
MILD = np.array([0.35, 0.55, 0.09, 0.01])
AMBIGUOUS = np.array([0.40, 0.35, 0.20, 0.05])
SEVERE = np.array([0.0, 0.02, 0.18, 0.80])
RUNAWAY = np.array([0.0, 0.0, 0.0, 1.0])


# ---------------------------------------------------------------------------
# expected loss arithmetic
# ---------------------------------------------------------------------------
def test_expected_loss_matches_the_definition():
    """E[loss](a) must equal the literal sum over states."""
    posterior = np.array([0.5, 0.3, 0.15, 0.05])
    losses = expected_losses(posterior)
    for i, _ in enumerate(P.ACTIONS):
        manual = sum(posterior[s] * P.LOSS_MATRIX[i, s] for s in range(P.N_STATES))
        assert losses[i] == pytest.approx(manual)


def test_expected_loss_rejects_a_mismatched_posterior():
    with pytest.raises(ValueError):
        expected_losses(np.array([0.5, 0.5]))


def test_expected_loss_normalises_an_unnormalised_posterior():
    a = expected_losses(np.array([2.0, 1.2, 0.6, 0.2]))
    b = expected_losses(np.array([2.0, 1.2, 0.6, 0.2]) / 4.0)
    assert np.allclose(a, b)


# ---------------------------------------------------------------------------
# the recommendation itself
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "posterior", [HEALTHY, MILD, AMBIGUOUS, SEVERE, RUNAWAY,
                  np.full(P.N_STATES, 0.25)]
)
def test_decision_returns_a_valid_action(posterior):
    d = recommend_action(posterior)
    assert d.recommended_action in P.ACTIONS
    assert set(d.expected_losses) == set(P.ACTIONS)
    assert len(d.evaluations) == len(P.ACTIONS)
    assert all(np.isfinite(v) for v in d.expected_losses.values())
    assert d.margin >= 0.0


def test_recommended_action_is_the_argmin():
    for posterior in (HEALTHY, MILD, AMBIGUOUS, SEVERE, RUNAWAY):
        d = recommend_action(posterior)
        best = min(d.expected_losses, key=d.expected_losses.get)
        assert d.recommended_action == best
        assert d.expected_losses[d.recommended_action] == pytest.approx(
            min(d.expected_losses.values())
        )


def test_severe_risk_prefers_an_aggressive_intervention():
    """With most of the mass on Thermal Runaway, containment must win."""
    d = recommend_action(RUNAWAY)
    assert d.recommended_action in ("Emergency Shutdown", "Isolate Affected Module")
    assert d.expected_losses["Continue Monitoring"] > d.expected_losses[d.recommended_action]


def test_emergency_shutdown_becomes_preferable_as_risk_grows():
    """Sweep the posterior from healthy to runaway and check the ordering flips."""
    losses_monitor, losses_shutdown = [], []
    for w in np.linspace(0.0, 1.0, 21):
        posterior = (1.0 - w) * HEALTHY + w * RUNAWAY
        d = recommend_action(posterior)
        losses_monitor.append(d.expected_losses["Continue Monitoring"])
        losses_shutdown.append(d.expected_losses["Emergency Shutdown"])

    assert losses_monitor[0] < losses_shutdown[0]     # healthy: monitoring wins
    assert losses_shutdown[-1] < losses_monitor[-1]   # runaway: shutdown wins
    # And the recommendation really does become Emergency Shutdown at the end.
    assert recommend_action(RUNAWAY).recommended_action == "Emergency Shutdown"


def test_low_risk_prefers_monitoring_or_a_backup_measurement():
    for posterior in (HEALTHY, np.array([0.92, 0.07, 0.01, 0.0])):
        d = recommend_action(posterior)
        assert d.recommended_action in (
            "Continue Monitoring", "Request Backup Measurement"
        )


def test_ambiguous_risk_does_not_jump_straight_to_shutdown():
    """Cheap information-gathering or mild mitigation should win when the
    posterior is genuinely uncertain."""
    d = recommend_action(AMBIGUOUS)
    assert d.recommended_action != "Emergency Shutdown"
    assert d.recommended_action in (
        "Continue Monitoring", "Request Backup Measurement",
        "Reduce Charging Current", "Increase Cooling",
    )


def test_reason_mentions_the_sensor_when_the_primary_is_untrusted():
    d = recommend_action(MILD, sensor_fault_probability=0.93)
    assert "trustworthy" in d.reason
    quiet = recommend_action(MILD, sensor_fault_probability=0.01)
    assert "trustworthy" not in quiet.reason


def test_action_evaluations_are_internally_consistent():
    d = recommend_action(SEVERE)
    for ev in d.evaluations:
        assert ev.expected_loss == pytest.approx(sum(ev.per_state_contribution.values()))
        assert 0.0 <= ev.intervention_effectiveness <= 1.0
        assert 0.0 <= ev.projected_dangerous_after <= 1.0
    assert d.ranked()[0].action == d.recommended_action


# ---------------------------------------------------------------------------
# threshold baseline and comparison
# ---------------------------------------------------------------------------
def test_threshold_baseline_levels():
    assert threshold_baseline(30.0).level == "Normal"
    assert threshold_baseline(30.0).triggered is False
    assert threshold_baseline(P.THRESHOLD_WARN_C + 1).level == "Warning"
    assert threshold_baseline(P.THRESHOLD_CRITICAL_C + 1).level == "Critical"
    assert threshold_baseline(P.THRESHOLD_CRITICAL_C + 1).action == "Emergency Shutdown"


def test_comparison_panel_reports_disagreement_on_a_faulty_sensor():
    """The whole point of the project, expressed as a test."""
    d = recommend_action(HEALTHY, sensor_fault_probability=0.99)
    comp = compare_with_threshold(92.0, d, sensor_fault_probability=0.99)
    assert comp["threshold_action"] == "Emergency Shutdown"
    assert comp["bayesian_action"] != "Emergency Shutdown"
    assert comp["agreement"] == "Disagree"
    assert "faulty" in comp["explanation"]


def test_comparison_panel_reports_agreement_on_a_genuine_runaway():
    d = recommend_action(RUNAWAY, sensor_fault_probability=0.02)
    comp = compare_with_threshold(95.0, d, sensor_fault_probability=0.02)
    assert comp["agreement"] == "Agree"


# ---------------------------------------------------------------------------
# full pipeline behaviour
# ---------------------------------------------------------------------------
def test_pipeline_produces_valid_output_for_every_scenario():
    for scenario in P.SCENARIOS:
        sim = simulate_pack(scenario, affected_cell=2, seed=3, n_steps=60)
        results = run_pipeline(sim.frame)
        assert set(results) == set(range(P.N_CELLS))
        for outs in results.values():
            assert len(outs) == 60
            for o in outs:
                assert o.posterior.sum() == pytest.approx(1.0, abs=1e-10)
                assert sum(o.cause.posterior.values()) == pytest.approx(1.0)
                assert 0.0 <= o.sensor.p_faulty <= 1.0
                assert o.decision.recommended_action in P.ACTIONS
                for h in P.FORECAST_HORIZONS_MIN:
                    assert 0.0 <= o.forecast.dangerous_by_horizon[float(h)] <= 1.0


def test_pipeline_does_not_shut_down_a_healthy_pack_with_a_broken_sensor():
    """End-to-end version of the demonstration case."""
    sim = simulate_pack("Temperature Sensor Fault", affected_cell=2, seed=42, n_steps=180)
    outs = run_pipeline(sim.frame)[2]
    last = outs[-1]

    assert last.reading["temp_primary"] > P.THRESHOLD_CRITICAL_C   # threshold would trip
    assert threshold_baseline(last.reading["temp_primary"]).action == "Emergency Shutdown"

    assert last.sensor.p_faulty > 0.9
    assert last.forecast.current_dangerous < 0.1
    assert last.decision.recommended_action != "Emergency Shutdown"


def test_pipeline_does_shut_down_a_genuine_runaway():
    """The safety-critical converse: the system must not be too sceptical."""
    sim = simulate_pack("Internal Short Circuit", affected_cell=2, seed=42, n_steps=180)
    last = run_pipeline(sim.frame)[2][-1]

    assert last.sensor.p_faulty < 0.3
    assert last.forecast.current_dangerous > 0.9
    assert last.decision.recommended_action in (
        "Emergency Shutdown", "Isolate Affected Module"
    )
    assert last.cause.top_cause == "Internal Short Circuit"


def test_pipeline_reset_clears_all_state():
    sim = simulate_pack("Internal Short Circuit", affected_cell=0, seed=9, n_steps=60)
    rows = sim.frame[sim.frame["cell"] == 0].sort_values("step")

    pipe = CellPipeline(cell=0)
    for _, row in rows.iterrows():
        pipe.step(row)
    assert len(pipe.history) == 60

    pipe.reset()
    assert pipe.history == []
    assert np.allclose(pipe.filter.posterior, P.INITIAL_STATE_PROBS)

    # Replaying must give bit-identical results after a reset.
    first = [o.posterior.copy() for o in
             [pipe.step(row) for _, row in rows.iterrows()]]
    pipe.reset()
    second = [o.posterior.copy() for o in
              [pipe.step(row) for _, row in rows.iterrows()]]
    assert all(np.allclose(a, b) for a, b in zip(first, second))
