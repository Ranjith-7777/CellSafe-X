"""Tests for the Phase-3 model-based/controlled-transition intervention
forecast (models/intervention.py) and its wiring into the decision engine."""

from __future__ import annotations

import numpy as np
import pytest

from config import model_parameters as P
from models.decision_engine import recommend_action
from models.intervention import (
    compare_interventions,
    controlled_transition_matrix,
    forecast_under_intervention,
    forecast_under_intervention_multi_horizon,
    intervention_summary,
)

STRENGTH_ORDER = [
    "Continue Monitoring",
    "Request Backup Measurement",
    "Reduce Charging Current",
    "Increase Cooling",
    "Isolate Affected Module",
    "Emergency Shutdown",
]

DANGEROUS_BELIEFS = [
    np.array([0.05, 0.85, 0.09, 0.01]),
    np.array([0.0, 0.15, 0.75, 0.10]),
    np.array([0.0, 0.0, 0.20, 0.80]),
]


# ---------------------------------------------------------------------------
# controlled transition matrices
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("action", P.ACTIONS)
def test_controlled_transition_matrix_is_valid(action):
    A = controlled_transition_matrix(action)
    assert A.shape == (P.N_STATES, P.N_STATES)
    assert np.allclose(A.sum(axis=1), 1.0)
    assert (A >= 0.0).all()
    assert (A <= 1.0).all()


def test_continue_monitoring_transition_matrix_equals_baseline():
    A = controlled_transition_matrix("Continue Monitoring")
    assert np.allclose(A, P.TRANSITION_MATRIX)


def test_request_backup_measurement_transition_matrix_equals_baseline():
    """Backup measurement is information-gathering, not a physical
    intervention - it must not alter the dynamics at all."""
    A = controlled_transition_matrix("Request Backup Measurement")
    assert np.allclose(A, P.TRANSITION_MATRIX)


def test_unknown_action_is_rejected():
    with pytest.raises(ValueError):
        controlled_transition_matrix("Not A Real Action")


# ---------------------------------------------------------------------------
# forecast API
# ---------------------------------------------------------------------------
def test_forecast_distributions_normalise():
    for belief in DANGEROUS_BELIEFS:
        for action in P.ACTIONS:
            cf = forecast_under_intervention(belief, action, horizon_min=15.0)
            assert cf.baseline_distribution.sum() == pytest.approx(1.0)
            assert cf.intervention_distribution.sum() == pytest.approx(1.0)
            assert (cf.baseline_distribution >= -1e-12).all()
            assert (cf.intervention_distribution >= -1e-12).all()


def test_forecast_does_not_mutate_the_input_posterior():
    belief = np.array([0.05, 0.85, 0.09, 0.01])
    original = belief.copy()
    forecast_under_intervention(belief, "Emergency Shutdown", horizon_min=15.0)
    assert np.array_equal(belief, original)


def test_forecast_is_deterministic():
    belief = [0.0, 0.15, 0.75, 0.10]
    a = forecast_under_intervention(belief, "Increase Cooling", horizon_min=15.0)
    b = forecast_under_intervention(belief, "Increase Cooling", horizon_min=15.0)
    assert np.array_equal(a.intervention_distribution, b.intervention_distribution)
    assert a.risk_reduction == b.risk_reduction


def test_continue_monitoring_has_zero_risk_reduction():
    for belief in DANGEROUS_BELIEFS:
        cf = forecast_under_intervention(belief, "Continue Monitoring", horizon_min=15.0)
        assert cf.risk_reduction == pytest.approx(0.0, abs=1e-9)
        assert cf.baseline_dangerous == pytest.approx(cf.intervention_dangerous, abs=1e-9)


def test_backup_measurement_has_no_fake_physical_risk_reduction():
    for belief in DANGEROUS_BELIEFS:
        cf = forecast_under_intervention(belief, "Request Backup Measurement", horizon_min=15.0)
        assert cf.risk_reduction == pytest.approx(0.0, abs=1e-9)


def test_multi_horizon_forecast_covers_the_standard_panel():
    belief = [0.0, 0.15, 0.75, 0.10]
    out = forecast_under_intervention_multi_horizon(belief, "Increase Cooling")
    assert set(out.keys()) == {float(h) for h in P.FORECAST_HORIZONS_MIN}
    for h, cf in out.items():
        assert cf.horizon_min == h
        assert cf.horizon_steps == P.horizon_steps(h)


def test_risk_reduction_matches_baseline_minus_intervention():
    belief = [0.0, 0.15, 0.75, 0.10]
    cf = forecast_under_intervention(belief, "Isolate Affected Module", horizon_min=15.0)
    assert cf.risk_reduction == pytest.approx(cf.baseline_dangerous - cf.intervention_dangerous)


# ---------------------------------------------------------------------------
# sanity invariants: stronger actions must not look riskier
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("belief", DANGEROUS_BELIEFS)
@pytest.mark.parametrize("horizon", P.FORECAST_HORIZONS_MIN)
def test_stronger_actions_never_predict_more_danger_than_weaker_ones(belief, horizon):
    results = {r.action: r for r in compare_interventions(belief, horizon_min=float(horizon))}
    dangerous = [results[a].intervention_dangerous for a in STRENGTH_ORDER]
    for i in range(len(dangerous) - 1):
        assert dangerous[i + 1] <= dangerous[i] + 1e-9, (
            f"{STRENGTH_ORDER[i + 1]} predicted more danger than {STRENGTH_ORDER[i]}"
        )


def test_emergency_shutdown_not_worse_than_continue_monitoring():
    for belief in DANGEROUS_BELIEFS:
        cm = forecast_under_intervention(belief, "Continue Monitoring", horizon_min=15.0)
        shutdown = forecast_under_intervention(belief, "Emergency Shutdown", horizon_min=15.0)
        assert shutdown.intervention_dangerous <= cm.intervention_dangerous + 1e-9


def test_increase_cooling_does_not_worsen_escalation_risk():
    for belief in DANGEROUS_BELIEFS:
        cf = forecast_under_intervention(belief, "Increase Cooling", horizon_min=15.0)
        assert cf.risk_reduction >= -1e-9


def test_healthy_state_dynamics_not_unrealistically_distorted():
    """From a near-certainly-healthy belief, baseline drift over 30 minutes is
    itself modest (the Healthy self-loop is 0.996/step, not 1.0), so no
    action should be able to manufacture a LARGE additional swing on top of
    that baseline drift - interventions matter for a pack that is already
    heating, not for a healthy one that barely needs them."""
    belief = np.array([0.999, 0.0009, 0.00005, 0.00005])
    baseline = forecast_under_intervention(belief, "Continue Monitoring", horizon_min=30.0)
    for action in P.ACTIONS:
        cf = forecast_under_intervention(belief, action, horizon_min=30.0)
        assert cf.intervention_distribution[P.S_HEALTHY] >= baseline.intervention_distribution[P.S_HEALTHY] - 1e-9


# ---------------------------------------------------------------------------
# comparison / summary: prediction vs decision stay separate
# ---------------------------------------------------------------------------
def test_compare_interventions_is_sorted_by_risk_reduction_descending():
    belief = [0.0, 0.15, 0.75, 0.10]
    ranking = compare_interventions(belief, horizon_min=15.0)
    reductions = [r.risk_reduction for r in ranking]
    assert reductions == sorted(reductions, reverse=True)


def test_intervention_summary_does_not_override_expected_loss_decision():
    """The risk-reduction ranking and the expected-loss decision are
    independent: the top of one ranking need not equal the top of the other,
    and intervention_summary must report BOTH without collapsing them into a
    single number."""
    belief = [0.0, 0.15, 0.75, 0.10]
    summary = intervention_summary(belief, horizon_min=15.0)
    decision = recommend_action(belief)
    assert summary.recommended_action == decision.recommended_action
    assert summary.expected_losses == decision.expected_losses
    assert summary.by_risk_reduction[0].action == "Emergency Shutdown"  # best risk reduction
    # Expected-loss minimiser is not required to be Emergency Shutdown - the
    # two rankings are allowed to (and typically do) disagree.


# ---------------------------------------------------------------------------
# decision engine wiring: no longer reads the deprecated static table
# ---------------------------------------------------------------------------
def test_action_evaluation_projected_dangerous_matches_controlled_forecast():
    belief = [0.0, 0.15, 0.75, 0.10]
    decision = recommend_action(belief)
    for ev in decision.evaluations:
        cf = forecast_under_intervention(belief, ev.action, horizon_min=P.DT_MINUTES)
        assert ev.projected_dangerous_after == pytest.approx(cf.intervention_dangerous)
        assert ev.intervention_effectiveness == pytest.approx(cf.relative_risk_reduction)


def test_decision_engine_evaluations_still_bounded():
    belief = [0.05, 0.85, 0.09, 0.01]
    decision = recommend_action(belief)
    for ev in decision.evaluations:
        assert 0.0 <= ev.intervention_effectiveness <= 1.0
        assert 0.0 <= ev.projected_dangerous_after <= 1.0
