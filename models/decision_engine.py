"""
Bayesian decision engine: minimum expected loss action selection.

For every candidate action a we compute

    ExpectedLoss(a) = SUM_state P(state | evidence) * Loss(a, state)

and recommend argmin_a ExpectedLoss(a).  The posterior comes from the Bayesian
filter, so the recommendation shifts continuously as evidence accumulates -
there is no threshold ladder anywhere in this module.

A separate, clearly-labelled "probabilistic intervention assessment" estimates
how much each action is expected to reduce the near-term dangerous-state
probability.  This is an assumed effectiveness model used for presentation only;
it is NOT formal causal counterfactual inference and is never used to override
the expected-loss ranking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from config import model_parameters as P
from models.risk_forecast import dangerous_probability, propagate


@dataclass
class ActionEvaluation:
    action: str
    expected_loss: float
    per_state_contribution: Dict[str, float]   # P(s) * L(a, s) for each state
    cost_note: str
    intervention_effectiveness: float
    projected_dangerous_after: float           # assessment only, see module docstring


@dataclass
class DecisionResult:
    recommended_action: str
    expected_losses: Dict[str, float]
    evaluations: List[ActionEvaluation]
    reason: str
    margin: float = 0.0                        # gap to the runner-up action
    posterior: np.ndarray = field(default_factory=lambda: np.zeros(P.N_STATES))

    def ranked(self) -> List[ActionEvaluation]:
        return sorted(self.evaluations, key=lambda e: e.expected_loss)


def expected_losses(posterior: Sequence[float], loss_matrix: np.ndarray | None = None) -> np.ndarray:
    """Vector of expected losses, one per action:  L @ p."""
    L = P.LOSS_MATRIX if loss_matrix is None else np.asarray(loss_matrix, dtype=float)
    p = np.asarray(posterior, dtype=float)
    if p.shape[0] != L.shape[1]:
        raise ValueError("posterior length must match the loss matrix state dimension")
    s = p.sum()
    if s <= 0:
        raise ValueError("posterior must have positive mass")
    p = p / s
    return L @ p


def recommend_action(
    posterior: Sequence[float],
    loss_matrix: np.ndarray | None = None,
    sensor_fault_probability: float | None = None,
) -> DecisionResult:
    """Choose the action with the lowest expected loss under the current belief.

    `sensor_fault_probability` is optional and does NOT change the arithmetic;
    it is only used to enrich the human-readable reason, so that a recommendation
    of "Request Backup Measurement" can say *why* more evidence is worth the
    delay.
    """
    L = P.LOSS_MATRIX if loss_matrix is None else np.asarray(loss_matrix, dtype=float)
    p = np.asarray(posterior, dtype=float)
    p = p / p.sum()

    losses = expected_losses(p, L)
    order = np.argsort(losses)
    best_idx = int(order[0])
    best_action = P.ACTIONS[best_idx]
    margin = float(losses[int(order[1])] - losses[best_idx]) if len(order) > 1 else 0.0

    danger = dangerous_probability(p)

    evaluations: List[ActionEvaluation] = []
    for i, action in enumerate(P.ACTIONS):
        contributions = {P.STATES[s]: float(p[s] * L[i, s]) for s in range(P.N_STATES)}
        eff = P.INTERVENTION_EFFECTIVENESS[action]
        # One-step-ahead dangerous probability if we do nothing, scaled down by
        # the assumed effectiveness of this action.
        baseline_next = dangerous_probability(propagate(p, 1))
        evaluations.append(
            ActionEvaluation(
                action=action,
                expected_loss=float(losses[i]),
                per_state_contribution=contributions,
                cost_note=P.ACTION_COST_NOTES[action],
                intervention_effectiveness=eff,
                projected_dangerous_after=float(baseline_next * (1.0 - eff)),
            )
        )

    reason = _build_reason(best_action, p, danger, margin, sensor_fault_probability)

    return DecisionResult(
        recommended_action=best_action,
        expected_losses={a: float(v) for a, v in zip(P.ACTIONS, losses)},
        evaluations=evaluations,
        reason=reason,
        margin=margin,
        posterior=p,
    )


def _build_reason(
    action: str,
    posterior: np.ndarray,
    danger: float,
    margin: float,
    sensor_fault_probability: float | None,
) -> str:
    dominant = P.STATES[int(np.argmax(posterior))]
    parts = [
        f"Posterior mass is concentrated on '{dominant}' "
        f"({posterior.max() * 100:.1f}%), with {danger * 100:.1f}% total probability on "
        f"the dangerous states."
    ]
    parts.append(
        f"'{action}' minimises expected loss; it beats the next-best action by "
        f"{margin:.2f} loss units."
    )
    if sensor_fault_probability is not None and sensor_fault_probability > 0.4:
        parts.append(
            f"The primary temperature sensor is only {(1 - sensor_fault_probability) * 100:.0f}% "
            f"likely to be trustworthy, so committing to an expensive intervention on its word "
            f"alone would be premature."
        )
    return " ".join(parts)


# ---------------------------------------------------------------------------
# threshold baseline for comparison
# ---------------------------------------------------------------------------
@dataclass
class ThresholdResult:
    level: str          # "Normal" | "Warning" | "Critical"
    action: str
    triggered: bool
    temperature_c: float
    note: str


def threshold_baseline(primary_temp_c: float) -> ThresholdResult:
    """The classical system we compare against: one fixed trip point on the
    primary temperature sensor, with no notion of evidence agreement.

    This is deliberately naive.  Its whole purpose is to show what happens when
    a single sensor is trusted unconditionally.
    """
    t = float(primary_temp_c)
    if t >= P.THRESHOLD_CRITICAL_C:
        return ThresholdResult(
            level="Critical",
            action="Emergency Shutdown",
            triggered=True,
            temperature_c=t,
            note=f"Primary sensor reads {t:.1f} C, above the {P.THRESHOLD_CRITICAL_C:.0f} C critical trip point.",
        )
    if t >= P.THRESHOLD_WARN_C:
        return ThresholdResult(
            level="Warning",
            action="Reduce Charging Current",
            triggered=True,
            temperature_c=t,
            note=f"Primary sensor reads {t:.1f} C, above the {P.THRESHOLD_WARN_C:.0f} C warning threshold.",
        )
    return ThresholdResult(
        level="Normal",
        action="Continue Monitoring",
        triggered=False,
        temperature_c=t,
        note=f"Primary sensor reads {t:.1f} C, below the {P.THRESHOLD_WARN_C:.0f} C warning threshold.",
    )


def compare_with_threshold(
    primary_temp_c: float,
    decision: DecisionResult,
    sensor_fault_probability: float,
) -> Dict[str, str]:
    """Side-by-side explanation of the two systems for the comparison panel."""
    base = threshold_baseline(primary_temp_c)
    agree = base.action == decision.recommended_action

    if agree:
        explanation = (
            "Both systems agree here. The evidence channels corroborate the primary "
            "temperature reading, so multi-evidence reasoning reaches the same conclusion "
            "as the fixed threshold - it simply reaches it with a calibrated probability "
            "attached instead of a binary trip."
        )
    elif sensor_fault_probability > 0.5:
        explanation = (
            f"They disagree because the threshold system trusts the primary sensor "
            f"unconditionally, while the Bayesian system assigns {sensor_fault_probability * 100:.1f}% "
            f"posterior probability to that sensor being faulty. The backup sensor, vent-gas, "
            f"voltage and neighbouring-cell channels all contradict the hot reading, so the "
            f"evidence is better explained by a broken sensor than by a hot cell."
        )
    elif base.triggered and not agree:
        explanation = (
            "The threshold has tripped on temperature alone. The Bayesian system weighs the "
            "full posterior over hidden states against the cost of each action and finds a "
            "cheaper action with lower expected loss - an over-reaction carries real cost too."
        )
    else:
        explanation = (
            "The temperature is still below the trip point, so the threshold system sees "
            "nothing at all. The Bayesian filter is already accumulating evidence from the "
            "heating rate, gas and voltage channels and recommends acting earlier."
        )

    return {
        "threshold_level": base.level,
        "threshold_action": base.action,
        "threshold_note": base.note,
        "bayesian_action": decision.recommended_action,
        "bayesian_note": decision.reason,
        "agreement": "Agree" if agree else "Disagree",
        "explanation": explanation,
    }
