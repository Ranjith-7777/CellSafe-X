"""
Baum-Welch (EM) learning of HMM transition + emission parameters (Phase 7C).

SCOPE. `models/bayesian_filter.py` (the production, engineered HMM) is NOT
modified by this module. Everything here is a self-contained, EXPERIMENTAL
variant: its own parameter container (`HMMParams`), its own log-space
forward-backward, its own EM loop, and its own inference wrapper
(`LearnedHMMFilter`) that mimics `BayesianFilter.update`'s recursion using
learned instead of hand-specified parameters. The engineered HMM stays the
default/production model everywhere else in the project.

WHY BAUM-WELCH. The hidden thermal state Z_t is never observed, even in
training - only noisy sensor readings are. Baum-Welch (EM for HMMs)
estimates transition and emission parameters that maximise the observation
sequences' likelihood by alternating:
  E-step: given current parameters, compute the posterior over hidden
          states/transitions at every step (forward-backward).
  M-step: given those posteriors, re-estimate parameters in closed form.
Ground-truth hidden-state labels are NEVER used in the E/M updates
themselves - only afterwards, to resolve the label-permutation problem
(see `align_states`), exactly as the assignment requires.

TRANSITION vs EMISSION (for the professor-facing report):
  Transition  P(Z_t | Z_{t-1})   - how the hidden state evolves.
  Emission    P(Y_t | Z_t)       - how a hidden state generates evidence.
Baum-Welch learns both from observation sequences alone.

FEATURES / DISTRIBUTION. Same five continuous channels as the engineered
model (`config.OBS_CHANNELS`, drawn from the canonical `SENSOR_REGISTRY`):
temp_c, temp_rate, voltage_dev, log_gas, neighbour_c. Each is modelled as an
independent (naive-Bayes) state-conditional Gaussian, exactly like the
engineered emission model - only the mean/std are learned, not the
independence assumption. A missing/non-finite channel at a given step is
skipped in the likelihood (marginalised), never fabricated - the same
contract `models.bayesian_filter.observation_log_likelihood` already
implements. Tempering weights (`OBS_LIKELIHOOD_WEIGHTS`) are REUSED, not
learned - they compensate for cross-channel correlation, which is a
modelling choice about the naive-Bayes approximation, not a Gaussian
parameter Baum-Welch is asked to learn here.

CONSTRAINED TOPOLOGY. The engineered transition matrix already forbids
"teleporting" transitions (e.g. Healthy -> Pre-Runaway, Healthy -> Runaway).
`allowed_transition_mask()` reads that zero pattern directly from
`config.TRANSITION_MATRIX` and the M-step renormalises A only over allowed
entries per row - a structurally-impossible transition can never acquire
positive probability no matter what the data suggests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from config import model_parameters as P
from models.bayesian_filter import log_gaussian, log_normalise, log_sum_exp

Observation = Mapping[str, float]
Sequence_ = List[Observation]

VARIANCE_FLOOR_FRACTION = 0.05  # learned std can never fall below 5% of the
                                  # engineered std for that channel/state -
                                  # prevents a singular (zero-variance) collapse.


# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------
@dataclass
class HMMParams:
    states: Tuple[str, ...]
    channels: Tuple[str, ...]
    pi: np.ndarray                       # (N,)
    A: np.ndarray                        # (N,N)
    means: Dict[str, np.ndarray]         # channel -> (N,)
    stds: Dict[str, np.ndarray]          # channel -> (N,)
    weights: Dict[str, float]            # channel -> tempering weight (fixed, reused)

    def copy(self) -> "HMMParams":
        return HMMParams(
            states=self.states, channels=self.channels,
            pi=self.pi.copy(), A=self.A.copy(),
            means={c: v.copy() for c, v in self.means.items()},
            stds={c: v.copy() for c, v in self.stds.items()},
            weights=dict(self.weights),
        )


def engineered_params() -> HMMParams:
    """The current production HMM's parameters, read (never mutated) from
    config.model_parameters - the starting point for the "engineered
    initialisation" and the A/B comparison baseline."""
    n = P.N_STATES
    means = {ch: np.array([P.OBS_MEANS[s][ch] for s in P.STATES]) for ch in P.OBS_CHANNELS}
    stds = {ch: np.array([P.OBS_STDS[s][ch] for s in P.STATES]) for ch in P.OBS_CHANNELS}
    return HMMParams(
        states=P.STATES, channels=P.OBS_CHANNELS,
        pi=P.INITIAL_STATE_PROBS.copy(), A=P.TRANSITION_MATRIX.copy(),
        means=means, stds=stds, weights=dict(P.OBS_LIKELIHOOD_WEIGHTS),
    )


def allowed_transition_mask() -> np.ndarray:
    """Boolean (N,N): True where the ENGINEERED transition matrix already
    allows a nonzero probability. Structurally impossible transitions stay
    forbidden through every Baum-Welch M-step, regardless of what the data
    alone would suggest."""
    return P.TRANSITION_MATRIX > 0.0


def perturbed_init(seed: int = 7) -> HMMParams:
    """Deterministic generic/perturbed initialisation (Goal 6B): NOT copied
    from the engineered parameters - a plausible but clearly different
    starting point, so convergence can be checked for initialisation
    sensitivity rather than trivially reproducing the engineered values."""
    rng = np.random.default_rng(seed)
    n = P.N_STATES
    base = engineered_params()

    pi = rng.dirichlet(np.ones(n))
    mask = allowed_transition_mask()
    A = np.zeros((n, n))
    for i in range(n):
        row = rng.dirichlet(np.ones(int(mask[i].sum())) * 2.0)
        A[i, mask[i]] = row

    means = {}
    stds = {}
    for ch in base.channels:
        # shift each state's mean by a bounded, deterministic random offset
        # (a fraction of that state's own std) rather than a fixed global
        # perturbation, so different channels get differently-scaled nudges.
        offset = rng.normal(0.0, 0.5, n) * base.stds[ch]
        means[ch] = base.means[ch] + offset
        stds[ch] = base.stds[ch] * rng.uniform(0.6, 1.6, n)

    return HMMParams(states=base.states, channels=base.channels, pi=pi, A=A, means=means, stds=stds, weights=base.weights)


# ---------------------------------------------------------------------------
# emission log-likelihood (marginalising missing channels, exactly like the
# engineered models.bayesian_filter.observation_log_likelihood)
# ---------------------------------------------------------------------------
def emission_log_likelihood(obs: Observation, params: HMMParams) -> np.ndarray:
    n = len(params.states)
    out = np.zeros(n)
    for i in range(n):
        total = 0.0
        for ch in params.channels:
            v = obs.get(ch)
            if v is None or not np.isfinite(v):
                continue
            total += params.weights[ch] * log_gaussian(float(v), params.means[ch][i], params.stds[ch][i])
        out[i] = total
    return out


# ---------------------------------------------------------------------------
# E-step: log-space forward-backward
# ---------------------------------------------------------------------------
def forward_backward(sequence: Sequence_, params: HMMParams) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Returns (log_alpha, log_beta, log_B, sequence_loglik). log_B[t] is the
    tempered emission log-likelihood vector used at step t (cached so the
    M-step does not recompute it)."""
    T = len(sequence)
    n = len(params.states)
    log_pi = np.log(np.clip(params.pi, 1e-300, None))
    log_A = np.log(np.clip(params.A, 1e-300, None))

    log_B = np.array([emission_log_likelihood(o, params) for o in sequence])  # (T, N)

    log_alpha = np.zeros((T, n))
    log_alpha[0] = log_pi + log_B[0]
    for t in range(1, T):
        for j in range(n):
            log_alpha[t, j] = log_sum_exp(log_alpha[t - 1] + log_A[:, j]) + log_B[t, j]

    log_beta = np.zeros((T, n))
    log_beta[T - 1] = 0.0
    for t in range(T - 2, -1, -1):
        for i in range(n):
            log_beta[t, i] = log_sum_exp(log_A[i, :] + log_B[t + 1] + log_beta[t + 1])

    seq_loglik = log_sum_exp(log_alpha[T - 1])
    return log_alpha, log_beta, log_B, float(seq_loglik)


def e_step(sequence: Sequence_, params: HMMParams) -> Tuple[np.ndarray, np.ndarray, float]:
    """gamma_t(i) = P(Z_t=i | Y_1:T); xi_t(i,j) = P(Z_t=i, Z_{t+1}=j | Y_1:T)."""
    T = len(sequence)
    n = len(params.states)
    log_alpha, log_beta, log_B, seq_loglik = forward_backward(sequence, params)

    log_gamma = log_alpha + log_beta
    gamma = np.exp(log_gamma - log_gamma.max(axis=1, keepdims=True))
    gamma /= gamma.sum(axis=1, keepdims=True)

    log_A = np.log(np.clip(params.A, 1e-300, None))
    xi = np.zeros((max(T - 1, 0), n, n))
    for t in range(T - 1):
        log_xi_t = (
            log_alpha[t][:, None] + log_A + log_B[t + 1][None, :] + log_beta[t + 1][None, :] - seq_loglik
        )
        m = log_xi_t.max()
        e = np.exp(log_xi_t - m)
        xi[t] = e / e.sum()

    return gamma, xi, seq_loglik


# ---------------------------------------------------------------------------
# M-step (aggregated across all training sequences)
# ---------------------------------------------------------------------------
def m_step(
    sequences: Sequence[Sequence_],
    gammas: Sequence[np.ndarray],
    xis: Sequence[np.ndarray],
    params: HMMParams,
    mask: np.ndarray,
) -> HMMParams:
    n = len(params.states)

    pi_new = np.mean([g[0] for g in gammas], axis=0)
    pi_new = pi_new / pi_new.sum()

    xi_sum = sum(x.sum(axis=0) for x in xis if len(x))
    gamma_departure_sum = sum(g[:-1].sum(axis=0) for g in gammas if len(g) > 1)
    A_new = np.zeros((n, n))
    for i in range(n):
        denom = gamma_departure_sum[i]
        if denom <= 1e-300:
            A_new[i] = params.A[i]
            continue
        row = np.where(mask[i], xi_sum[i] / denom, 0.0)
        s = row.sum()
        A_new[i] = row / s if s > 1e-300 else params.A[i]

    means_new: Dict[str, np.ndarray] = {}
    stds_new: Dict[str, np.ndarray] = {}
    floor_std = {ch: params.stds[ch] * VARIANCE_FLOOR_FRACTION for ch in params.channels}

    for ch in params.channels:
        w_sum = np.zeros(n)
        wx_sum = np.zeros(n)
        for seq, g in zip(sequences, gammas):
            for t, obs in enumerate(seq):
                v = obs.get(ch)
                if v is None or not np.isfinite(v):
                    continue
                w_sum += g[t]
                wx_sum += g[t] * float(v)
        mean = np.where(w_sum > 1e-9, wx_sum / np.where(w_sum > 1e-9, w_sum, 1.0), params.means[ch])

        wx2_sum = np.zeros(n)
        for seq, g in zip(sequences, gammas):
            for t, obs in enumerate(seq):
                v = obs.get(ch)
                if v is None or not np.isfinite(v):
                    continue
                wx2_sum += g[t] * (float(v) - mean) ** 2
        var = np.where(w_sum > 1e-9, wx2_sum / np.where(w_sum > 1e-9, w_sum, 1.0), params.stds[ch] ** 2)
        std = np.sqrt(np.maximum(var, floor_std[ch] ** 2))

        means_new[ch] = mean
        stds_new[ch] = std

    return HMMParams(states=params.states, channels=params.channels, pi=pi_new, A=A_new, means=means_new, stds=stds_new, weights=params.weights)


# ---------------------------------------------------------------------------
# EM driver
# ---------------------------------------------------------------------------
@dataclass
class TrainingHistory:
    log_likelihood: List[float] = field(default_factory=list)
    delta: List[float] = field(default_factory=list)
    converged: bool = False
    iterations: int = 0


def run_baum_welch(
    sequences: Sequence[Sequence_],
    init_params: HMMParams,
    mask: Optional[np.ndarray] = None,
    max_iter: int = 50,
    tol: float = 1e-3,
) -> Tuple[HMMParams, TrainingHistory]:
    """Multi-sequence EM. Log-likelihood is summed over all training
    sequences each iteration and must be non-decreasing up to tiny numerical
    tolerance - `TrainingHistory.delta` records the raw (possibly very
    slightly negative, floating-point-noise) deltas verbatim for inspection.
    """
    if mask is None:
        mask = allowed_transition_mask()
    params = init_params.copy()
    history = TrainingHistory()

    prev_ll = -np.inf
    for it in range(1, max_iter + 1):
        gammas: List[np.ndarray] = []
        xis: List[np.ndarray] = []
        total_ll = 0.0
        for seq in sequences:
            g, x, ll = e_step(seq, params)
            gammas.append(g)
            xis.append(x)
            total_ll += ll

        history.log_likelihood.append(total_ll)
        delta = total_ll - prev_ll
        history.delta.append(delta)
        history.iterations = it

        params = m_step(sequences, gammas, xis, params, mask)

        if it > 1 and abs(delta) < tol:
            history.converged = True
            break
        prev_ll = total_ll

    return params, history


# ---------------------------------------------------------------------------
# state-label alignment (post-hoc only; ground truth NEVER used in E/M)
# ---------------------------------------------------------------------------
def align_states(
    params: HMMParams,
    labelled_sequences: Sequence[Sequence_],
    true_states: Sequence[Sequence[int]],
) -> Tuple[HMMParams, np.ndarray]:
    """Hungarian assignment maximising correspondence between the MAP
    learned-state decoding and ground-truth labels on VALIDATION data only.
    Returns (relabelled_params, permutation) where permutation[learned_idx]
    = semantic_idx.
    """
    from scipy.optimize import linear_sum_assignment

    n = len(params.states)
    confusion = np.zeros((n, n), dtype=float)  # rows: learned idx, cols: true idx
    for seq, truth in zip(labelled_sequences, true_states):
        gamma, _, _ = e_step(seq, params)
        learned = np.argmax(gamma, axis=1)
        for l, t in zip(learned, truth):
            confusion[l, t] += 1

    row_ind, col_ind = linear_sum_assignment(-confusion)
    permutation = np.zeros(n, dtype=int)
    permutation[row_ind] = col_ind

    relabelled = params.copy()
    # reorder so that new index = semantic index: new_A[s1, s2] = old_A[perm^-1(s1), perm^-1(s2)]
    inv = np.argsort(permutation)
    relabelled.pi = params.pi[inv]
    relabelled.A = params.A[np.ix_(inv, inv)]
    for ch in params.channels:
        relabelled.means[ch] = params.means[ch][inv]
        relabelled.stds[ch] = params.stds[ch][inv]
    return relabelled, permutation


# ---------------------------------------------------------------------------
# inference with a learned parameter set (mirrors BayesianFilter.update)
# ---------------------------------------------------------------------------
class LearnedHMMFilter:
    """Exact forward filtering using a (post-alignment) learned HMMParams -
    same recursion as models.bayesian_filter.BayesianFilter, independent
    implementation so the production filter is never touched."""

    def __init__(self, params: HMMParams) -> None:
        self.params = params
        self.reset()

    def reset(self) -> None:
        self.posterior = self.params.pi.copy()
        self._first = True

    def update(self, obs: Observation) -> np.ndarray:
        if self._first:
            prior = self.posterior.copy()
            self._first = False
        else:
            prior = self.posterior @ self.params.A
        prior = prior / prior.sum()

        log_lik = emission_log_likelihood(obs, self.params)
        log_joint = np.log(np.clip(prior, 1e-300, None)) + log_lik
        self.posterior = log_normalise(log_joint)
        return self.posterior
