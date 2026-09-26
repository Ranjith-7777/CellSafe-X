"""Tests for Phase-7C Baum-Welch (EM) learning of HMM transition + emission
parameters (models/baum_welch.py)."""

from __future__ import annotations

import numpy as np
import pytest

from config import model_parameters as P
from data.battery_simulator import simulate_pack
from models import CellPipeline
from models.baum_welch import (
    HMMParams,
    LearnedHMMFilter,
    align_states,
    allowed_transition_mask,
    e_step,
    engineered_params,
    forward_backward,
    m_step,
    perturbed_init,
    run_baum_welch,
)
from models.bayesian_filter import BayesianFilter, log_sum_exp


def _toy_sequences(n_seqs=3, length=20, seed=0):
    rng = np.random.default_rng(seed)
    seqs = []
    for _ in range(n_seqs):
        seqs.append([
            {"temp_c": float(30 + rng.normal(0, 4)), "temp_rate": float(rng.normal(0, 0.4))}
            for _ in range(length)
        ])
    return seqs


# ---------------------------------------------------------------------------
# forward/backward consistency
# ---------------------------------------------------------------------------
def test_forward_and_backward_give_the_same_sequence_loglikelihood():
    params = engineered_params()
    seq = _toy_sequences(1, 15)[0]
    log_alpha, log_beta, log_B, ll_forward = forward_backward(seq, params)
    # log P(Y) can also be recovered from alpha_t * beta_t for ANY t
    for t in [0, 5, 14]:
        ll_t = log_sum_exp(log_alpha[t] + log_beta[t])
        assert ll_t == pytest.approx(ll_forward, abs=1e-6)


def test_gamma_rows_normalise():
    params = engineered_params()
    seq = _toy_sequences(1, 25)[0]
    gamma, xi, ll = e_step(seq, params)
    assert np.allclose(gamma.sum(axis=1), 1.0, atol=1e-9)
    assert (gamma >= -1e-12).all()


def test_xi_normalises_per_timestep():
    params = engineered_params()
    seq = _toy_sequences(1, 25)[0]
    gamma, xi, ll = e_step(seq, params)
    sums = xi.sum(axis=(1, 2))
    assert np.allclose(sums, 1.0, atol=1e-9)


# ---------------------------------------------------------------------------
# M-step invariants
# ---------------------------------------------------------------------------
def test_transition_rows_normalise_after_m_step():
    params = engineered_params()
    mask = allowed_transition_mask()
    seqs = _toy_sequences(4, 20)
    gammas, xis = [], []
    for s in seqs:
        g, x, _ = e_step(s, params)
        gammas.append(g)
        xis.append(x)
    updated = m_step(seqs, gammas, xis, params, mask)
    assert np.allclose(updated.A.sum(axis=1), 1.0, atol=1e-9)


def test_emission_variances_stay_positive_after_m_step():
    params = engineered_params()
    mask = allowed_transition_mask()
    seqs = _toy_sequences(4, 20)
    gammas, xis = [], []
    for s in seqs:
        g, x, _ = e_step(s, params)
        gammas.append(g)
        xis.append(x)
    updated = m_step(seqs, gammas, xis, params, mask)
    for ch in updated.channels:
        assert (updated.stds[ch] > 0).all()


def test_allowed_transition_mask_matches_engineered_zero_pattern():
    mask = allowed_transition_mask()
    assert np.array_equal(mask, P.TRANSITION_MATRIX > 0.0)
    # explicit structurally-impossible transitions
    assert not mask[P.S_HEALTHY, P.S_PRE]
    assert not mask[P.S_HEALTHY, P.S_RUNAWAY]


def test_m_step_never_puts_mass_on_disallowed_transitions():
    params = engineered_params()
    mask = allowed_transition_mask()
    seqs = _toy_sequences(4, 20)
    gammas, xis = [], []
    for s in seqs:
        g, x, _ = e_step(s, params)
        gammas.append(g)
        xis.append(x)
    updated = m_step(seqs, gammas, xis, params, mask)
    assert np.allclose(updated.A[~mask], 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# EM loop: convergence, determinism
# ---------------------------------------------------------------------------
def test_log_likelihood_is_non_decreasing_within_tolerance():
    seqs = _toy_sequences(5, 30, seed=1)
    _, history = run_baum_welch(seqs, perturbed_init(seed=3), max_iter=15, tol=1e-4)
    for d in history.delta[1:]:
        assert d > -1e-6, f"log-likelihood decreased by {d}"


def test_training_is_deterministic_with_a_fixed_seed():
    seqs = _toy_sequences(4, 25, seed=2)
    p1, h1 = run_baum_welch(seqs, perturbed_init(seed=9), max_iter=10, tol=1e-4)
    p2, h2 = run_baum_welch(seqs, perturbed_init(seed=9), max_iter=10, tol=1e-4)
    assert np.array_equal(p1.A, p2.A)
    assert h1.log_likelihood == h2.log_likelihood


def test_perturbed_init_differs_from_engineered_init():
    eng = engineered_params()
    pert = perturbed_init(seed=7)
    assert not np.allclose(eng.A, pert.A)


def test_two_initializations_converge_to_comparable_log_likelihood():
    """Not a strict equality requirement (different local optima are
    possible in general), but on this well-behaved toy problem both should
    land close together - a basic initialisation-sensitivity check."""
    seqs = _toy_sequences(6, 30, seed=4)
    _, h_eng = run_baum_welch(seqs, engineered_params(), max_iter=25, tol=1e-3)
    _, h_pert = run_baum_welch(seqs, perturbed_init(seed=7), max_iter=25, tol=1e-3)
    assert abs(h_eng.log_likelihood[-1] - h_pert.log_likelihood[-1]) < 0.05 * abs(h_eng.log_likelihood[-1])


# ---------------------------------------------------------------------------
# state-label alignment (ground truth NEVER used inside E/M)
# ---------------------------------------------------------------------------
def test_align_states_is_deterministic():
    seqs = _toy_sequences(3, 20, seed=5)
    params, _ = run_baum_welch(seqs, perturbed_init(seed=1), max_iter=5, tol=1e-2)
    truths = [[0] * 20 for _ in seqs]
    a1, perm1 = align_states(params, seqs, truths)
    a2, perm2 = align_states(params, seqs, truths)
    assert np.array_equal(perm1, perm2)
    assert np.array_equal(a1.A, a2.A)


def test_align_states_returns_a_valid_permutation():
    seqs = _toy_sequences(3, 20, seed=6)
    params, _ = run_baum_welch(seqs, perturbed_init(seed=1), max_iter=5, tol=1e-2)
    truths = [[i % 4 for i in range(20)] for _ in seqs]
    _, perm = align_states(params, seqs, truths)
    assert sorted(perm.tolist()) == [0, 1, 2, 3]


def test_ground_truth_not_referenced_inside_e_or_m_step():
    import inspect

    import models.baum_welch as bw

    for fn in (bw.e_step, bw.m_step, bw.forward_backward, bw.run_baum_welch):
        src = inspect.getsource(fn)
        assert "true_state" not in src
        assert "truth" not in src


# ---------------------------------------------------------------------------
# missing-feature handling
# ---------------------------------------------------------------------------
def test_missing_channel_is_marginalised_not_fabricated():
    params = engineered_params()
    full = {"temp_c": 32.0, "temp_rate": 0.1, "voltage_dev": 0.02, "log_gas": 1.2, "neighbour_c": 31.0}
    partial = {"temp_c": 32.0, "temp_rate": 0.1}  # voltage_dev/log_gas/neighbour_c missing
    from models.baum_welch import emission_log_likelihood

    ll_full = emission_log_likelihood(full, params)
    ll_partial = emission_log_likelihood(partial, params)
    assert not np.allclose(ll_full, ll_partial)  # fewer channels -> different (smaller-magnitude) likelihood
    assert np.isfinite(ll_partial).all()


def test_nan_channel_is_treated_as_missing():
    params = engineered_params()
    from models.baum_welch import emission_log_likelihood

    a = emission_log_likelihood({"temp_c": 32.0, "temp_rate": float("nan")}, params)
    b = emission_log_likelihood({"temp_c": 32.0}, params)
    assert np.array_equal(a, b)


def test_em_handles_sequences_with_missing_channels():
    seqs = [
        [{"temp_c": 32.0 + i * 0.1} for i in range(15)],  # temp_rate missing throughout
        [{"temp_c": 33.0 + i * 0.1, "temp_rate": 0.1} for i in range(15)],
    ]
    params, history = run_baum_welch(seqs, perturbed_init(seed=2), max_iter=8, tol=1e-2)
    assert np.isfinite(params.A).all()
    assert history.log_likelihood[-1] > history.log_likelihood[0] - 1e-6


# ---------------------------------------------------------------------------
# inference with the learned model
# ---------------------------------------------------------------------------
def test_learned_hmm_filter_produces_normalised_posteriors():
    params = engineered_params()  # stand-in "learned" params for this test
    f = LearnedHMMFilter(params)
    seq = _toy_sequences(1, 10)[0]
    for obs in seq:
        p = f.update(obs)
        assert p.sum() == pytest.approx(1.0, abs=1e-9)
        assert (p >= 0).all()


def test_learned_hmm_filter_can_run_on_a_real_simulated_trajectory():
    sim = simulate_pack(scenario="Cooling-System Failure", affected_cell=2, seed=20240, n_steps=20)
    frame = sim.frame[sim.frame["cell"] == 2].sort_values("step")
    pipe = CellPipeline(cell=2, temperature_fusion="legacy")
    params = engineered_params()
    f = LearnedHMMFilter(params)
    for _, row in frame.iterrows():
        out = pipe.step(row)
        obs = {"temp_c": out.fused_temp_c, "temp_rate": out.temp_rate}
        p = f.update(obs)
        assert p.sum() == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# production HMM untouched
# ---------------------------------------------------------------------------
def test_engineered_params_match_config_exactly():
    params = engineered_params()
    assert np.array_equal(params.A, P.TRANSITION_MATRIX)
    assert np.array_equal(params.pi, P.INITIAL_STATE_PROBS)
    for ch in P.OBS_CHANNELS:
        assert np.array_equal(params.means[ch], np.array([P.OBS_MEANS[s][ch] for s in P.STATES]))


def test_production_bayesian_filter_unaffected_by_baum_welch_module():
    """Importing/using models.baum_welch must not mutate
    config.model_parameters or BayesianFilter's default behaviour."""
    before = P.TRANSITION_MATRIX.copy()
    _ = run_baum_welch(_toy_sequences(2, 10), perturbed_init(seed=1), max_iter=3)
    assert np.array_equal(P.TRANSITION_MATRIX, before)

    f = BayesianFilter()
    step = f.update({"temp_c": 32.0, "temp_rate": 0.05})
    assert step.posterior.sum() == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# train/validation/test trajectory-level separation (Goal 2 of the phase)
# ---------------------------------------------------------------------------
def test_train_validation_test_seed_ranges_are_disjoint():
    from evaluation.evaluate_baum_welch import (
        N_TEST_SEEDS,
        N_TRAIN_SEEDS,
        N_VAL_SEEDS,
        TEST_SEED_BASE,
        TRAIN_SEED_BASE,
        VAL_SEED_BASE,
    )

    train_seeds = set(range(TRAIN_SEED_BASE, TRAIN_SEED_BASE + N_TRAIN_SEEDS))
    val_seeds = set(range(VAL_SEED_BASE, VAL_SEED_BASE + N_VAL_SEEDS))
    test_seeds = set(range(TEST_SEED_BASE, TEST_SEED_BASE + N_TEST_SEEDS))
    assert train_seeds.isdisjoint(val_seeds)
    assert train_seeds.isdisjoint(test_seeds)
    assert val_seeds.isdisjoint(test_seeds)
