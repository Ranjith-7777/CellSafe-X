"""Tests for Phase-7B core-pipeline integration of overlapping-sensor
fusion (models.CellPipeline's `temperature_fusion` modes) and the A/B
evaluation path."""

from __future__ import annotations

import numpy as np
import pytest

from data.battery_simulator import simulate_pack
from models import CellPipeline, run_pipeline


def _frame(scenario="Cooling-System Failure", cell=2, seed=20240, n_steps=30):
    sim = simulate_pack(scenario=scenario, affected_cell=cell, seed=seed, n_steps=n_steps)
    return sim.frame[sim.frame["cell"] == cell].sort_values("step")


# ---------------------------------------------------------------------------
# default behaviour is unchanged (frozen baseline)
# ---------------------------------------------------------------------------
def test_default_temperature_fusion_is_legacy():
    pipe = CellPipeline(cell=2)
    assert pipe.temperature_fusion == "legacy"


def test_legacy_mode_never_populates_sensor_fusion_detail():
    frame = _frame()
    pipe = CellPipeline(cell=2, temperature_fusion="legacy")
    for _, row in frame.iterrows():
        out = pipe.step(row)
    assert out.sensor_fusion is None


def test_unknown_fusion_mode_is_rejected():
    with pytest.raises(ValueError):
        CellPipeline(cell=2, temperature_fusion="not_a_real_mode")


# ---------------------------------------------------------------------------
# overlapping mode integration
# ---------------------------------------------------------------------------
def test_overlapping_mode_populates_sensor_fusion_detail_with_all_three_sensors():
    frame = _frame()
    pipe = CellPipeline(cell=2, temperature_fusion="overlapping")
    out = None
    for _, row in frame.iterrows():
        out = pipe.step(row)
    assert out.sensor_fusion is not None
    assert set(out.sensor_fusion.readings) == {"primary", "surface", "backup"}


def test_overlapping_mode_posterior_still_normalises():
    frame = _frame()
    pipe = CellPipeline(cell=2, temperature_fusion="overlapping")
    for _, row in frame.iterrows():
        out = pipe.step(row)
        assert out.posterior.sum() == pytest.approx(1.0, abs=1e-9)
        assert (out.posterior >= 0).all()


def test_overlapping_mode_root_cause_and_decision_still_run():
    """sensor_reliability/root-cause/decision are unchanged in both modes -
    only the temp_c value differs."""
    frame = _frame()
    pipe = CellPipeline(cell=2, temperature_fusion="overlapping")
    out = None
    for _, row in frame.iterrows():
        out = pipe.step(row)
    assert out.cause.top_cause in out.cause.posterior
    assert out.decision.recommended_action


# ---------------------------------------------------------------------------
# A/B reproducibility
# ---------------------------------------------------------------------------
def test_run_pipeline_default_matches_explicit_legacy():
    frame = _frame()
    a = run_pipeline(frame)
    b = run_pipeline(frame, temperature_fusion="legacy")
    for cell in a:
        for oa, ob in zip(a[cell], b[cell]):
            assert oa.fused_temp_c == ob.fused_temp_c


def test_run_pipeline_overlapping_is_deterministic():
    frame = _frame()
    a = run_pipeline(frame, temperature_fusion="overlapping")
    b = run_pipeline(frame, temperature_fusion="overlapping")
    for cell in a:
        for oa, ob in zip(a[cell], b[cell]):
            assert oa.fused_temp_c == ob.fused_temp_c
            assert np.array_equal(oa.posterior, ob.posterior)


def test_legacy_and_overlapping_can_disagree_on_fused_temperature():
    """Sanity check that the two modes are actually wired to different
    fusion logic (not accidentally identical)."""
    frame = _frame()
    legacy = run_pipeline(frame, temperature_fusion="legacy")[2]
    overlap = run_pipeline(frame, temperature_fusion="overlapping")[2]
    diffs = [abs(a.fused_temp_c - b.fused_temp_c) for a, b in zip(legacy, overlap)]
    assert max(diffs) > 0.0


def test_legacy_mode_temp_surface_presence_does_not_change_fused_value():
    """Regression guard: temp_surface being present in the row must not
    perturb the legacy (primary/backup-only) fused temperature, whether or
    not the row happens to carry the extra column."""
    frame = _frame()
    frame_no_surface = frame.drop(columns=["temp_surface"])

    def _final_fused(fr):
        pipe = CellPipeline(cell=2, temperature_fusion="legacy")
        out = None
        for _, row in fr.iterrows():
            out = pipe.step(row)
        return out.fused_temp_c

    assert _final_fused(frame) == pytest.approx(_final_fused(frame_no_surface))
