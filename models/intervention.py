"""
Model-based intervention forecasting (Phase 3).

SCIENTIFIC SCOPE - READ BEFORE USING
-------------------------------------
CellSafe-X's hidden-state model is a manually-specified HMM/DBN: the
transition matrix, emissions and loss matrix are modelling assumptions, not
parameters estimated from interventional or randomised data. This module
therefore does NOT implement Pearlian do(X) causal identification, and
nothing here should be described as a "true counterfactual effect" or
"proven causal effect".

What it DOES implement is a controlled-transition forecast: for each action
we define an explicit alternative transition matrix A(action), derived from
the baseline TRANSITION_MATRIX by two documented multipliers (see
config/model_parameters.py :: INTERVENTION_TRANSITION_EFFECTS), and ask

    P_future(Z_{t+k} | Y_{1:t}, intervention = a) = b_t @ A(a)^k

i.e. "what would the k-step-ahead belief be if the system evolved under this
controlled dynamics instead of the baseline dynamics, starting from the
current filtered posterior". This is forecasting under an assumed model of
controlled dynamics - a standard and useful tool, but an assumption, not an
experimentally established intervention efficacy.

Two things this module explicitly does NOT do:
  * It never modifies `BayesianFilter`'s recursive evidence-filtering
    posterior. Counterfactual forecasts run forward from the CURRENT
    posterior; the filter itself is untouched.
  * It never overrides `models.decision_engine.recommend_action`, which
    continues to choose the minimum-expected-loss action from the ordinary
    (baseline) posterior. Risk reduction and expected loss answer different
    questions - see `intervention_summary` below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Sequence

import numpy as np

from config import model_parameters as P
from models.risk_forecast import dangerous_probability, propagate

DEFAULT_HORIZON_MIN: float = float(P.FORECAST_HORIZONS_MIN[0])  # "short" horizon


# ---------------------------------------------------------------------------
# controlled transition matrices
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def controlled_transition_matrix(action: str) -> np.ndarray:
    """A(action): TRANSITION_MATRIX with escalation/recovery entries scaled.

    Off-diagonal entries moving to a MORE dangerous state (column > row) are
    scaled by `escalation_multiplier`; entries moving to a LESS dangerous
    state (column < row) are scaled by `recovery_multiplier`. The diagonal is
    then recomputed as `1 - sum(off-diagonal)` so every row still sums to
    exactly 1 by construction - never renormalised by division, so this can
    never silently absorb an invalid (negative or >1) probability.
    """
    if action not in P.INTERVENTION_TRANSITION_EFFECTS:
        raise ValueError(f"unknown action {action!r}; expected one of {P.ACTIONS}")
    eff = P.INTERVENTION_TRANSITION_EFFECTS[action]
    esc, rec = float(eff["escalation_multiplier"]), float(eff["recovery_multiplier"])

    A0 = P.TRANSITION_MATRIX
    n = A0.shape[0]
    A = A0.copy()
    for i in range(n):
        for j in range(n):
            if j > i:
                A[i, j] = A0[i, j] * esc
            elif j < i:
                A[i, j] = A0[i, j] * rec
        off_diag_sum = float(A[i].sum() - A[i, i])
        A[i, i] = 1.0 - off_diag_sum

    if (A < -1e-9).any() or (A > 1.0 + 1e-9).any():
        raise ValueError(f"controlled transition matrix for {action!r} left [0,1]: {A}")
    A = np.clip(A, 0.0, 1.0)
    row_sums = A.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-9):
        raise ValueError(f"controlled transition matrix for {action!r} rows do not sum to 1: {row_sums}")
    return A


# ---------------------------------------------------------------------------
# counterfactual forecast API
# ---------------------------------------------------------------------------
@dataclass
class InterventionForecast:
    action: str
    horizon_min: float
    horizon_steps: int
    baseline_distribution: np.ndarray
    intervention_distribution: np.ndarray
    baseline_dangerous: float
    intervention_dangerous: float
    baseline_pre_runaway: float
    intervention_pre_runaway: float
    baseline_runaway: float
    intervention_runaway: float
    risk_reduction: float                # baseline_dangerous - intervention_dangerous, >= 0 by design
    relative_risk_reduction: float        # risk_reduction / baseline_dangerous, 0 if baseline is ~0


def forecast_under_intervention(
    posterior: Sequence[float],
    action: str,
    horizon_min: float = DEFAULT_HORIZON_MIN,
) -> InterventionForecast:
    """Counterfactual k-step-ahead forecast for one action from `posterior`.

    `posterior` is read-only here: `propagate()` copies its input, so the
    caller's array is never mutated, and the SAME baseline propagation (under
    the unmodified TRANSITION_MATRIX) is used for every action so risk
    reductions are directly comparable.
    """
    b = np.asarray(posterior, dtype=float)
    b = b / b.sum()

    steps = P.horizon_steps(horizon_min)
    baseline_dist = propagate(b, steps)  # baseline dynamics, no intervention
    intervention_dist = propagate(b, steps, transition=controlled_transition_matrix(action))

    baseline_dangerous = dangerous_probability(baseline_dist)
    intervention_dangerous = dangerous_probability(intervention_dist)
    risk_reduction = float(baseline_dangerous - intervention_dangerous)
    relative = float(risk_reduction / baseline_dangerous) if baseline_dangerous > 1e-9 else 0.0

    return InterventionForecast(
        action=action,
        horizon_min=float(horizon_min),
        horizon_steps=steps,
        baseline_distribution=baseline_dist,
        intervention_distribution=intervention_dist,
        baseline_dangerous=float(baseline_dangerous),
        intervention_dangerous=float(intervention_dangerous),
        baseline_pre_runaway=float(baseline_dist[P.S_PRE]),
        intervention_pre_runaway=float(intervention_dist[P.S_PRE]),
        baseline_runaway=float(baseline_dist[P.S_RUNAWAY]),
        intervention_runaway=float(intervention_dist[P.S_RUNAWAY]),
        risk_reduction=risk_reduction,
        relative_risk_reduction=max(0.0, min(1.0, relative)),
    )


def forecast_under_intervention_multi_horizon(
    posterior: Sequence[float],
    action: str,
    horizons_min: Sequence[float] = P.FORECAST_HORIZONS_MIN,
) -> Dict[float, InterventionForecast]:
    """The same forecast at several horizons (e.g. the project's standard
    5 / 15 / 30-minute panel)."""
    return {float(h): forecast_under_intervention(posterior, action, float(h)) for h in horizons_min}


def compare_interventions(
    posterior: Sequence[float],
    horizon_min: float = DEFAULT_HORIZON_MIN,
    actions: Sequence[str] = P.ACTIONS,
) -> List[InterventionForecast]:
    """All actions' counterfactual forecasts from the same posterior, ranked
    by risk reduction (largest first). This is a PREDICTION ranking only - it
    answers "what reduces future danger the most", not "what should we do"
    (that remains models.decision_engine.recommend_action, which also weighs
    operational cost). See `intervention_summary` to get both together without
    conflating them."""
    results = [forecast_under_intervention(posterior, a, horizon_min) for a in actions]
    return sorted(results, key=lambda r: r.risk_reduction, reverse=True)


@dataclass
class InterventionSummary:
    """Prediction (risk reduction ranking) and decision (expected-loss
    ranking) reported side by side, explicitly not merged into one score."""

    horizon_min: float
    by_risk_reduction: List[InterventionForecast] = field(default_factory=list)
    recommended_action: str = ""
    expected_losses: Dict[str, float] = field(default_factory=dict)


def intervention_summary(
    posterior: Sequence[float], horizon_min: float = DEFAULT_HORIZON_MIN
) -> InterventionSummary:
    """Convenience wrapper combining the risk-reduction ranking with the
    existing (unmodified) minimum-expected-loss decision, for callers that
    want both without re-deriving either. Importing decision_engine here
    (rather than the reverse) keeps intervention forecasting a pure add-on
    that decision_engine does not depend on.
    """
    from models.decision_engine import recommend_action  # local import: avoid a cycle

    ranking = compare_interventions(posterior, horizon_min)
    decision = recommend_action(posterior)
    return InterventionSummary(
        horizon_min=float(horizon_min),
        by_risk_reduction=ranking,
        recommended_action=decision.recommended_action,
        expected_losses=decision.expected_losses,
    )
