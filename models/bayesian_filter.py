"""
Exact Bayesian (HMM / one-slice Dynamic Bayesian Network) filtering.

The model is a first-order DBN over a single cell:

    Z_{t-1} ---> Z_t ---> Z_{t+1}        hidden thermal state
                  |
                  v
                 Y_t                      sensor observation vector

Online filtering recursion implemented here:

    predict :  P(Z_t | Y_{1:t-1}) = sum_{z} P(Z_t | Z_{t-1}=z) P(Z_{t-1}=z | Y_{1:t-1})
    update  :  P(Z_t | Y_{1:t})  = P(Y_t | Z_t) * P(Z_t | Y_{1:t-1}) / evidence

i.e. exactly

    P(Z_t | Y_1:t)  proportional-to  P(Y_t | Z_t) * SUM_z(t-1) P(Z_t | Z_t-1) P(Z_t-1 | Y_1:t-1)

Everything is computed in log space and normalised with a log-sum-exp so the
posterior is numerically stable and always sums to exactly one (to float
precision) even after thousands of steps.

No approximate inference is used: with four hidden states the exact recursion is
a 4x4 matrix-vector product, so there is nothing to approximate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping

import numpy as np

from config import model_parameters as P

_LOG_2PI = float(np.log(2.0 * np.pi))


# ---------------------------------------------------------------------------
# numerical helpers
# ---------------------------------------------------------------------------
def log_gaussian(x: float, mu: float, sigma: float) -> float:
    """Log density of N(mu, sigma^2) evaluated at x."""
    z = (x - mu) / sigma
    return -0.5 * (z * z + _LOG_2PI) - float(np.log(sigma))


def log_normalise(log_vec: np.ndarray) -> np.ndarray:
    """Exponentiate and normalise a vector of log-weights (log-sum-exp trick).

    Returns a proper probability vector summing to 1.  If every entry
    underflows to -inf (numerically impossible evidence) we fall back to the
    uniform distribution rather than producing NaNs.
    """
    log_vec = np.asarray(log_vec, dtype=float)
    m = np.max(log_vec)
    if not np.isfinite(m):
        return np.full(log_vec.shape, 1.0 / log_vec.size)
    shifted = np.exp(log_vec - m)
    total = shifted.sum()
    if total <= 0.0 or not np.isfinite(total):
        return np.full(log_vec.shape, 1.0 / log_vec.size)
    return shifted / total


def log_sum_exp(log_vec: np.ndarray) -> float:
    """Stable log(sum(exp(log_vec)))."""
    log_vec = np.asarray(log_vec, dtype=float)
    m = np.max(log_vec)
    if not np.isfinite(m):
        return float("-inf")
    return float(m + np.log(np.exp(log_vec - m).sum()))


# ---------------------------------------------------------------------------
# observation model
# ---------------------------------------------------------------------------
def observation_log_likelihood(obs: Mapping[str, float]) -> np.ndarray:
    """log P(Y_t | Z_t = s) for every hidden state s.

    `obs` must provide the channels listed in P.OBS_CHANNELS.  Channels that are
    missing or non-finite are simply skipped, which is the correct Bayesian
    treatment of a missing observation (it contributes no evidence).

    Each channel's log density is multiplied by a tempering weight < 1 (see
    P.OBS_LIKELIHOOD_WEIGHTS) because the channels are correlated in reality and
    a raw naive-Bayes product would be over-confident.
    """
    out = np.zeros(P.N_STATES, dtype=float)
    for i, state in enumerate(P.STATES):
        total = 0.0
        for ch in P.OBS_CHANNELS:
            value = obs.get(ch, None)
            if value is None or not np.isfinite(value):
                continue
            total += P.OBS_LIKELIHOOD_WEIGHTS[ch] * log_gaussian(
                float(value), P.OBS_MEANS[state][ch], P.OBS_STDS[state][ch]
            )
        out[i] = total
    return out


def observation_likelihood_normalised(obs: Mapping[str, float]) -> np.ndarray:
    """The emission term rescaled to sum to 1, purely for display.

    This is *not* used inside the recursion (the recursion needs the unnormalised
    likelihood), but it lets the dashboard show "how much each state is
    favoured by this observation alone".
    """
    return log_normalise(observation_log_likelihood(obs))


# ---------------------------------------------------------------------------
# the filter
# ---------------------------------------------------------------------------
@dataclass
class FilterStep:
    """Everything produced by one application of the recursion, kept so that the
    Model Explanation tab can display prior -> likelihood -> posterior."""

    prior: np.ndarray            # P(Z_t | Y_{1:t-1})  (after the transition step)
    log_likelihood: np.ndarray   # log P(Y_t | Z_t)    (unnormalised, tempered)
    likelihood_norm: np.ndarray  # likelihood rescaled to sum to 1, for display
    posterior: np.ndarray        # P(Z_t | Y_{1:t})
    log_evidence: float          # log P(Y_t | Y_{1:t-1}), the normalising constant


class BayesianFilter:
    """Exact forward filter for the four-state thermal DBN.

    Usage (online):
        f = BayesianFilter()
        for obs in stream:
            step = f.update(obs)
            print(step.posterior)
    """

    def __init__(
        self,
        initial: np.ndarray | None = None,
        transition: np.ndarray | None = None,
    ) -> None:
        self.initial = np.array(P.INITIAL_STATE_PROBS if initial is None else initial, dtype=float)
        self.transition = np.array(P.TRANSITION_MATRIX if transition is None else transition, dtype=float)
        if abs(self.initial.sum() - 1.0) > 1e-9:
            raise ValueError("initial distribution must sum to 1")
        if not np.allclose(self.transition.sum(axis=1), 1.0):
            raise ValueError("transition matrix rows must sum to 1")
        self.reset()

    # -- state management ---------------------------------------------------
    def reset(self) -> None:
        """Return the belief to the prior and forget the history."""
        self.posterior: np.ndarray = self.initial.copy()
        self.n_updates: int = 0
        self.total_log_evidence: float = 0.0
        self._first: bool = True

    # -- the two halves of the recursion ------------------------------------
    def predict(self, belief: np.ndarray | None = None, steps: int = 1) -> np.ndarray:
        """Push a belief `steps` steps into the future through the transition model.

        P(Z_{t+k} | Y_{1:t}) = P(Z_t | Y_{1:t}) @ A^k
        """
        if steps < 0:
            raise ValueError("steps must be non-negative")
        b = self.posterior if belief is None else np.asarray(belief, dtype=float)
        b = b / b.sum()
        for _ in range(steps):
            b = b @ self.transition
        return b / b.sum()

    def update(self, obs: Mapping[str, float]) -> FilterStep:
        """One full recursion step: transition-predict, then Bayes-update on Y_t."""
        # --- prediction half -------------------------------------------------
        # At the very first observation the prior is P(Z_0) itself; the sensor
        # reading at t=0 is an observation of the initial state, so we do not
        # apply a transition before it.
        if self._first:
            prior = self.posterior.copy()
            self._first = False
        else:
            prior = self.posterior @ self.transition

        prior = prior / prior.sum()

        # --- update half -----------------------------------------------------
        log_lik = observation_log_likelihood(obs)
        log_joint = np.log(np.clip(prior, 1e-300, None)) + log_lik

        log_evidence = log_sum_exp(log_joint)
        posterior = log_normalise(log_joint)

        self.posterior = posterior
        self.n_updates += 1
        if np.isfinite(log_evidence):
            self.total_log_evidence += log_evidence

        return FilterStep(
            prior=prior,
            log_likelihood=log_lik,
            likelihood_norm=observation_likelihood_normalised(obs),
            posterior=posterior,
            log_evidence=float(log_evidence),
        )

    # -- batch convenience --------------------------------------------------
    def run(self, observations: Iterable[Mapping[str, float]]) -> List[FilterStep]:
        """Filter a whole sequence from the current belief, returning every step."""
        return [self.update(obs) for obs in observations]


# ---------------------------------------------------------------------------
# post-hoc posterior calibration (temperature scaling)
# ---------------------------------------------------------------------------
def calibrated_posterior(posterior: np.ndarray, temperature: float) -> np.ndarray:
    """Softens (temperature > 1) or sharpens (temperature < 1) a posterior for
    REPORTING PURPOSES ONLY.

    This is deliberately NOT applied inside `BayesianFilter.update`: the belief
    fed back into next step's `predict()` must stay the exact, un-rescaled
    Bayesian posterior, otherwise the recursion itself would be silently
    changed and would no longer be exact filtering. Temperature scaling is
    applied only when a probability is about to be *displayed or scored*,
    exactly like temperature scaling on a classifier's logits: it reshapes the
    output distribution without touching the model's internal state.

    ``calibrated = normalise(posterior ** (1 / temperature))``

    temperature == 1.0 is the identity (returns `posterior` unchanged, up to
    renormalisation). temperature > 1.0 flattens an over-confident posterior;
    temperature < 1.0 sharpens an under-confident one. The MAP state (argmax)
    is invariant to any temperature > 0, so accuracy is never affected - only
    Brier score, NLL and calibration error are.
    """
    posterior = np.asarray(posterior, dtype=float)
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if temperature == 1.0:
        return posterior / posterior.sum()
    log_p = np.log(np.clip(posterior, 1e-300, None)) / temperature
    return log_normalise(log_p)
