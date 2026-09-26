"""Tests for the synthetic simulator's ground-truth labelling, in particular
the Phase-2 fix requiring a sustained heating-rate excursion (not a single
noisy step) before the label escalates a band."""

from __future__ import annotations

import numpy as np

from config import model_parameters as P
from data.battery_simulator import (
    LABEL_ABNORMAL_C,
    LABEL_PRE_C,
    LABEL_RATE_ESCALATE,
    RATE_ESCALATE_MIN_STEPS,
    _label_state,
    simulate_pack,
)


# ---------------------------------------------------------------------------
# _label_state: sustained-rate requirement
# ---------------------------------------------------------------------------
def test_single_step_rate_spike_does_not_escalate_the_label():
    """A one-off noisy rate reading (streak == 1) must NOT bump the band."""
    temp = LABEL_ABNORMAL_C + 1.0  # sits in the Abnormal band on temperature alone
    state = _label_state(temp, LABEL_RATE_ESCALATE + 1.0, rate_escalate_streak=1)
    assert state == P.S_ABNORMAL


def test_sustained_rate_excursion_does_escalate_the_label():
    """Once the elevated rate has persisted for RATE_ESCALATE_MIN_STEPS, the
    label escalates exactly as before."""
    temp = LABEL_ABNORMAL_C + 1.0
    state = _label_state(
        temp, LABEL_RATE_ESCALATE + 1.0, rate_escalate_streak=RATE_ESCALATE_MIN_STEPS
    )
    assert state == P.S_ABNORMAL + 1


def test_rate_escalation_never_overrides_a_direct_temperature_crossing():
    """Rate is only a tie-breaker inside a band; it must never matter once the
    temperature itself has crossed a threshold."""
    assert _label_state(LABEL_PRE_C + 5.0, 0.0, rate_escalate_streak=0) == P.S_PRE
    assert _label_state(LABEL_PRE_C + 5.0, 0.0, rate_escalate_streak=99) == P.S_PRE


def test_rate_escalation_never_promotes_past_pre_runaway():
    temp = LABEL_ABNORMAL_C + 1.0
    state = _label_state(temp, 50.0, rate_escalate_streak=999)
    assert state <= P.S_PRE


def test_label_state_default_streak_is_zero_and_does_not_escalate():
    """Calling without a streak argument (as no in-repo caller does any more,
    but the default must stay safe) must not spuriously escalate."""
    temp = LABEL_ABNORMAL_C + 1.0
    assert _label_state(temp, LABEL_RATE_ESCALATE + 5.0) == P.S_ABNORMAL


# ---------------------------------------------------------------------------
# regression guard: whole-run label stability
# ---------------------------------------------------------------------------
def _one_step_reversals(true_state: np.ndarray) -> int:
    """Count A, B, A one-step flicker patterns in a state sequence."""
    count = 0
    for i in range(1, len(true_state) - 1):
        if true_state[i] != true_state[i - 1] and true_state[i + 1] == true_state[i - 1]:
            count += 1
    return count


def test_external_heat_exposure_label_flicker_is_bounded():
    """Regression guard for the Phase-2 fix: before it, this scenario/seed
    produced dozens of one-step ground-truth reversals purely from process
    noise on the rate channel. After requiring a sustained excursion, flicker
    must stay low. A deterministic seed keeps this reproducible."""
    sim = simulate_pack(scenario="External Heat Exposure", affected_cell=2, seed=20240, n_steps=180)
    for cell in range(P.N_CELLS):
        ts = (
            sim.frame[sim.frame["cell"] == cell]
            .sort_values("step")["true_state"]
            .to_numpy()
        )
        assert _one_step_reversals(ts) <= 3, f"cell {cell} flickered {_one_step_reversals(ts)} times"


def test_dangerous_state_segments_are_not_dominated_by_single_step_flicker():
    """Across a full scenario run, most dangerous-state (Pre-Runaway+) segments
    should last more than one step. This is a direct, reproducible check of
    the property the Phase-2 audit quantified (median segment length rose
    from 1 step to several steps after the fix)."""
    sim = simulate_pack(scenario="External Heat Exposure", affected_cell=2, seed=20240, n_steps=180)
    single_step = 0
    total = 0
    for cell in range(P.N_CELLS):
        ts = (
            sim.frame[sim.frame["cell"] == cell]
            .sort_values("step")["true_state"]
            .to_numpy()
        )
        dangerous = ts >= P.S_PRE
        idx = np.where(dangerous)[0]
        if len(idx) == 0:
            continue
        splits = np.where(np.diff(idx) > 1)[0]
        starts = [idx[0]] + [idx[s + 1] for s in splits]
        ends = [idx[s] for s in splits] + [idx[-1]]
        for s, e in zip(starts, ends):
            total += 1
            if e == s:
                single_step += 1
    assert total > 0
    assert single_step / total < 0.5


# ---------------------------------------------------------------------------
# ground truth stays internally consistent
# ---------------------------------------------------------------------------
def test_true_state_never_below_healthy_for_ambient_temperature():
    sim = simulate_pack(scenario="Normal Operation", affected_cell=0, seed=1, n_steps=60)
    assert (sim.frame["true_state"] >= P.S_HEALTHY).all()


def test_internal_short_circuit_still_reaches_thermal_runaway():
    """The sustained-rate fix must not weaken a genuinely fast, sustained
    escalation - Internal Short Circuit heats for many consecutive steps, so
    it must still reach Thermal Runaway ground truth."""
    sim = simulate_pack(scenario="Internal Short Circuit", affected_cell=2, seed=20240, n_steps=180)
    g = sim.frame[sim.frame["cell"] == 2]
    assert (g["true_state"] == P.S_RUNAWAY).any()
