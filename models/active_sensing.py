"""
Active sensing / Value of Information (Phase 4, Part B).

Answers "which additional measurement is worth requesting next", using the
SAME emission likelihood model the Bayesian filter already uses - no new
sensor channel or likelihood is invented here (see
config.model_parameters.ACTIVE_SENSING_CANDIDATE_CHANNELS and its docstring
for exactly why "current" and "cooling_eff" are excluded).

Expected Information Gain (nats, natural log throughout):

    EIG(channel) = H(P(Z)) - E_{y ~ P(Y_channel)}[H(P(Z | y))]

H is Shannon entropy of the 4-state posterior. The expectation over possible
observations y is estimated by Monte Carlo ancestral sampling from the
current posterior's predictive mixture: draw a hidden state s' ~ P(Z), then
y ~ N(mean_{s'}, std_{s'}) for that channel (this is exactly P(Y_channel), the
posterior-weighted predictive distribution), then Bayes-update the posterior
using only that channel's tempered likelihood and measure its entropy. The
sample count and seed are fixed constants
(config.ACTIVE_SENSING_MC_SAMPLES/_MC_SEED), so every call is deterministic.

Expected Value of Information (decision loss units, same scale as
models.decision_engine.expected_losses):

    EVI(channel) = min_a E[Loss(a)] now - E_y[min_a E[Loss(a) | y]]

By a standard Bayesian decision-theory result ("information never hurts" a
loss-minimising decision maker), EVI >= 0 in expectation; a small negative
value can appear only from Monte Carlo noise at finite sample size.

Neither figure is a claim that the measurement itself changes anything
physical - see `models.intervention` and the Phase 3 disclosure that Request
Backup Measurement carries zero physical hazard reduction. This module is
what makes that action's non-physical value legible: how much would OBSERVING
more evidence be worth, not how much would ACTING change the pack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from config import model_parameters as P
from models.bayesian_filter import log_gaussian, log_normalise
from models.risk_forecast import dangerous_probability


# ---------------------------------------------------------------------------
# entropy
# ---------------------------------------------------------------------------
def posterior_entropy(belief: Sequence[float]) -> float:
    """Shannon entropy of a probability vector, in NATS. 0 = certain,
    ln(N_STATES) = maximally uncertain (uniform)."""
    b = np.asarray(belief, dtype=float)
    b = b / b.sum()
    mask = b > 0
    return float(-np.sum(b[mask] * np.log(b[mask])))


def _single_channel_update(belief: np.ndarray, channel: str, y: float) -> np.ndarray:
    """Bayes-update `belief` using ONE extra reading of `channel`, with the
    same tempered log-likelihood the filter uses for that channel."""
    w = P.OBS_LIKELIHOOD_WEIGHTS[channel]
    log_lik = np.array(
        [w * log_gaussian(y, P.OBS_MEANS[state][channel], P.OBS_STDS[state][channel]) for state in P.STATES]
    )
    log_post = np.log(np.clip(belief, 1e-300, None)) + log_lik
    return log_normalise(log_post)


def _predictive_samples(
    belief: np.ndarray, channel: str, n_samples: int, seed: int
) -> np.ndarray:
    """n_samples updated posteriors, from y drawn from the predictive mixture
    P(y) = sum_s belief_s * N(mean_s, std_s) via ancestral sampling."""
    rng = np.random.default_rng(seed)
    state_idx = rng.choice(P.N_STATES, size=n_samples, p=belief)
    out = np.empty((n_samples, P.N_STATES), dtype=float)
    for k, s in enumerate(state_idx):
        state = P.STATES[s]
        y = rng.normal(P.OBS_MEANS[state][channel], P.OBS_STDS[state][channel])
        out[k] = _single_channel_update(belief, channel, y)
    return out


# ---------------------------------------------------------------------------
# information gain
# ---------------------------------------------------------------------------
@dataclass
class InformationGainResult:
    channel: str
    sensor_family: str                      # Goal 4: config.OBS_CHANNEL_TO_SENSOR_FAMILY[channel]
    current_entropy: float                 # nats
    expected_posterior_entropy: float       # nats
    expected_information_gain: float        # nats, = current - expected_posterior
    relative_information_score: float       # EIG / current_entropy, 0 if current_entropy ~ 0
    n_samples: int
    seed: int


def expected_information_gain(
    belief: Sequence[float],
    channel: str,
    n_samples: int = P.ACTIVE_SENSING_MC_SAMPLES,
    seed: int = P.ACTIVE_SENSING_MC_SEED,
) -> InformationGainResult:
    if channel not in P.ACTIVE_SENSING_CANDIDATE_CHANNELS:
        raise ValueError(
            f"{channel!r} has no state-conditional likelihood in the hidden-state "
            f"filter; supported channels are {P.ACTIVE_SENSING_CANDIDATE_CHANNELS}"
        )
    b = np.asarray(belief, dtype=float)
    b = b / b.sum()
    h0 = posterior_entropy(b)
    posts = _predictive_samples(b, channel, n_samples, seed)
    expected_h = float(np.mean([posterior_entropy(p) for p in posts]))
    eig = h0 - expected_h
    relative = eig / h0 if h0 > 1e-12 else 0.0
    return InformationGainResult(
        channel=channel,
        sensor_family=P.OBS_CHANNEL_TO_SENSOR_FAMILY.get(channel, "unknown"),
        current_entropy=h0,
        expected_posterior_entropy=expected_h,
        expected_information_gain=eig,
        relative_information_score=float(max(0.0, min(1.0, relative))),
        n_samples=n_samples,
        seed=seed,
    )


def rank_information_gain(
    belief: Sequence[float], channels: Sequence[str] = P.ACTIVE_SENSING_CANDIDATE_CHANNELS
) -> List[InformationGainResult]:
    """All candidate channels' EIG, best (largest) first."""
    results = [expected_information_gain(belief, c) for c in channels]
    return sorted(results, key=lambda r: r.expected_information_gain, reverse=True)


# ---------------------------------------------------------------------------
# expected value of information (decision-loss-aware)
# ---------------------------------------------------------------------------
@dataclass
class ValueOfInformationResult:
    channel: str
    current_min_expected_loss: float
    expected_min_expected_loss_after: float
    expected_value_of_information: float   # >= 0 in expectation; may be tiny/negative from MC noise
    n_samples: int
    seed: int


def expected_value_of_information(
    belief: Sequence[float],
    channel: str,
    n_samples: int = P.ACTIVE_SENSING_MC_SAMPLES,
    seed: int = P.ACTIVE_SENSING_MC_SEED,
) -> ValueOfInformationResult:
    from models.decision_engine import expected_losses  # local import: avoid a cycle

    b = np.asarray(belief, dtype=float)
    b = b / b.sum()
    current_min = float(expected_losses(b).min())
    posts = _predictive_samples(b, channel, n_samples, seed)
    mean_min_after = float(np.mean([expected_losses(p).min() for p in posts]))
    evi = current_min - mean_min_after
    return ValueOfInformationResult(
        channel=channel,
        current_min_expected_loss=current_min,
        expected_min_expected_loss_after=mean_min_after,
        expected_value_of_information=evi,
        n_samples=n_samples,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# gated recommendation (Backup Measurement integration)
# ---------------------------------------------------------------------------
@dataclass
class ActiveSensingRecommendation:
    should_request_measurement: bool
    reasons: List[str]
    best_channel: str
    information_gain: List[InformationGainResult]
    value_of_information: ValueOfInformationResult
    could_plausibly_change_action: bool
    current_confidence: float
    current_dangerous_probability: float


def recommend_measurement(belief: Sequence[float]) -> ActiveSensingRecommendation:
    """Gated recommendation for whether "Request Backup Measurement" is
    actually worth its delay, and which channel it should target.

    Gating (config.ACTIVE_SENSING_* - centralised, not UI-only magic numbers):
      * posterior already confident (max prob >= CONFIDENCE_THRESHOLD)       -> skip
      * P(dangerous) already emergency-level (>= EMERGENCY_DANGEROUS_P)      -> skip
        (delaying action to gather more evidence is treated as unsafe)
      * even the best channel's EIG is negligible (< MIN_EIG_NATS)           -> skip
    Any one of these is sufficient to recommend against measuring; none of
    them claims the measurement itself would physically reduce hazard.
    """
    b = np.asarray(belief, dtype=float)
    b = b / b.sum()

    ranking = rank_information_gain(b)
    best = ranking[0]
    voi = expected_value_of_information(b, best.channel)

    confidence = float(b.max())
    danger = dangerous_probability(b)

    reasons: List[str] = []
    should_measure = True
    if confidence >= P.ACTIVE_SENSING_CONFIDENCE_THRESHOLD:
        should_measure = False
        reasons.append(
            f"posterior is already confident ({confidence:.3f} >= "
            f"{P.ACTIVE_SENSING_CONFIDENCE_THRESHOLD:.2f}); the state is not ambiguous."
        )
    if danger >= P.ACTIVE_SENSING_EMERGENCY_DANGEROUS_P:
        should_measure = False
        reasons.append(
            f"P(dangerous) = {danger:.3f} is at/above the emergency threshold "
            f"({P.ACTIVE_SENSING_EMERGENCY_DANGEROUS_P:.2f}); delaying action to "
            f"gather more evidence is treated as unsafe."
        )
    if best.expected_information_gain < P.ACTIVE_SENSING_MIN_EIG_NATS:
        should_measure = False
        reasons.append(
            f"best candidate channel ({best.channel}) has negligible expected "
            f"information gain ({best.expected_information_gain:.4f} nats < "
            f"{P.ACTIVE_SENSING_MIN_EIG_NATS:.2f})."
        )
    if should_measure:
        reasons.append(
            f"posterior is ambiguous ({confidence:.3f} confidence, "
            f"{posterior_entropy(b):.3f} nats entropy) and {best.channel} offers "
            f"{best.expected_information_gain:.4f} nats of expected information gain."
        )

    # A positive EVI implies the posterior-conditional decision sometimes
    # differs from the current one (otherwise the expected minimum loss after
    # observing could not fall below the current minimum). A tiny positive
    # value from Monte Carlo noise is not treated as a plausible flip.
    could_change_action = voi.expected_value_of_information > 1e-4

    return ActiveSensingRecommendation(
        should_request_measurement=should_measure,
        reasons=reasons,
        best_channel=best.channel,
        information_gain=ranking,
        value_of_information=voi,
        could_plausibly_change_action=could_change_action,
        current_confidence=confidence,
        current_dangerous_probability=float(danger),
    )
