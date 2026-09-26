"""Tests for Phase-7A overlapping-sensor reliability-aware fusion
(models/sensor_fusion.py) and the simulator's third sensor channel."""

from __future__ import annotations

import numpy as np
import pytest

from config import model_parameters as P
from data.battery_simulator import simulate_pack
from models.sensor_fusion import fuse_overlapping_sensors, select_temperature_evidence


# ---------------------------------------------------------------------------
# healthy agreement / outlier / stuck / bias behaviour
# ---------------------------------------------------------------------------
def test_healthy_agreement_gives_high_reliability_to_all_sensors():
    r = fuse_overlapping_sensors({"primary": 72.0, "surface": 69.0, "backup": 71.0})
    assert all(rel > 0.85 for rel in r.reliability.values())
    assert min(69.0, 71.0, 72.0) <= r.fused_temp_c <= max(69.0, 71.0, 72.0)


def test_one_outlier_sensor_loses_reliability_and_does_not_dominate_fusion():
    r = fuse_overlapping_sensors({"primary": 110.0, "surface": 49.0, "backup": 51.0})
    assert r.reliability["primary"] < 0.1
    assert r.reliability["surface"] > 0.9
    assert r.reliability["backup"] > 0.9
    # fused estimate must sit near the agreeing pair, not be dragged toward 110
    assert abs(r.fused_temp_c - 50.0) < 5.0


def test_stuck_sensor_loses_reliability_once_it_diverges():
    live = fuse_overlapping_sensors({"primary": 40.0, "surface": 41.0, "backup": 40.5})
    stuck = fuse_overlapping_sensors({"primary": 40.0, "surface": 80.0, "backup": 40.5})
    assert stuck.reliability["surface"] < live.reliability["surface"]


def test_positive_and_negative_bias_are_both_detected():
    pos = fuse_overlapping_sensors({"primary": 95.0, "surface": 70.0, "backup": 71.0})
    neg = fuse_overlapping_sensors({"primary": 45.0, "surface": 70.0, "backup": 71.0})
    assert pos.reliability["primary"] < 0.5
    assert neg.reliability["primary"] < 0.5


# ---------------------------------------------------------------------------
# reliability normalisation / fusion boundedness / numerical stability
# ---------------------------------------------------------------------------
def test_reliability_values_are_bounded_in_zero_one():
    for readings in (
        {"primary": 72.0, "surface": 69.0, "backup": 71.0},
        {"primary": 200.0, "surface": 20.0, "backup": 21.0},
        {"primary": 20.0, "surface": 20.0, "backup": 20.0},
    ):
        r = fuse_overlapping_sensors(readings)
        for v in r.reliability.values():
            assert 0.0 <= v <= 1.0


def test_subset_posterior_normalises_to_one():
    r = fuse_overlapping_sensors({"primary": 72.0, "surface": 69.0, "backup": 71.0})
    assert sum(r.subset_posterior.values()) == pytest.approx(1.0, abs=1e-9)


def test_fused_temperature_is_bounded_by_the_readings_convex_hull_extremes():
    readings = {"primary": 110.0, "surface": 49.0, "backup": 51.0}
    r = fuse_overlapping_sensors(readings)
    lo, hi = min(readings.values()), max(readings.values())
    assert lo - 1e-6 <= r.fused_temp_c <= hi + 1e-6


def test_single_sensor_input_does_not_crash_and_is_trivially_reliable():
    r = fuse_overlapping_sensors({"primary": 55.0})
    assert r.fused_temp_c == pytest.approx(55.0)
    assert r.reliability["primary"] == pytest.approx(P.SENSOR_PRIOR["reliable"])


def test_extreme_disagreement_is_numerically_stable():
    r = fuse_overlapping_sensors({"primary": 5000.0, "surface": 20.0, "backup": 21.0})
    assert np.isfinite(r.fused_temp_c)
    assert all(np.isfinite(v) for v in r.reliability.values())


def test_fusion_is_deterministic():
    a = fuse_overlapping_sensors({"primary": 72.0, "surface": 69.0, "backup": 71.0})
    b = fuse_overlapping_sensors({"primary": 72.0, "surface": 69.0, "backup": 71.0})
    assert a.fused_temp_c == b.fused_temp_c
    assert a.reliability == b.reliability


def test_rejects_empty_readings():
    with pytest.raises(ValueError):
        fuse_overlapping_sensors({})


# ---------------------------------------------------------------------------
# no simulator fault-label leakage
# ---------------------------------------------------------------------------
def test_fusion_function_signature_has_no_fault_label_parameter():
    """`fault_sigma` (the broad-residual scale, a config constant) is fine;
    a ground-truth leak would look like `sensor_faulty_true` or `true_state`."""
    import inspect

    from models.sensor_fusion import fuse_overlapping_sensors as f

    params = set(inspect.signature(f).parameters)
    assert not any(p.lower() in {"sensor_faulty_true", "true_state", "is_faulty"} for p in params)


def test_fusion_module_never_imports_the_simulator():
    import inspect

    import models.sensor_fusion as sf

    source = inspect.getsource(sf)
    assert "battery_simulator" not in source
    assert "sensor_faulty_true" not in source


# ---------------------------------------------------------------------------
# simulator's third overlapping sensor channel
# ---------------------------------------------------------------------------
def test_simulator_produces_a_temp_surface_channel_for_every_scenario():
    for scenario in P.SCENARIOS:
        sim = simulate_pack(scenario=scenario, affected_cell=2, seed=20240, n_steps=30)
        assert "temp_surface" in sim.frame.columns
        assert np.isfinite(sim.frame["temp_surface"]).all()


def test_temp_surface_is_not_identical_to_primary_or_backup():
    sim = simulate_pack(scenario="Cooling-System Failure", affected_cell=2, seed=20240, n_steps=60)
    frame = sim.frame[sim.frame["cell"] == 2]
    assert not np.allclose(frame["temp_surface"], frame["temp_primary"])
    assert not np.allclose(frame["temp_surface"], frame["temp_backup"])


def test_adding_temp_surface_does_not_change_existing_channels_determinism():
    """Regression guard: the new sensor's RNG draws must not shift the
    pre-existing channels' random stream (frozen-baseline reproducibility)."""
    sim = simulate_pack(scenario="Internal Short Circuit", affected_cell=2, seed=20240, n_steps=60)
    frame = sim.frame[sim.frame["cell"] == 2].reset_index(drop=True)
    # Known frozen-baseline value from Phases 1-6 (temp_primary at step 0,
    # cell 2, seed 20240, before temp_surface existed).
    assert frame.loc[0, "temp_primary"] == pytest.approx(32.64594973445087, abs=1e-6)


# ---------------------------------------------------------------------------
# Phase 7B Goal 6: missing-sensor robustness
# ---------------------------------------------------------------------------
FULL_READING = {"temp_primary": 45.0, "temp_surface": 43.0, "temp_backup": 44.0}


def test_select_temperature_evidence_with_all_three_present():
    fused, detail = select_temperature_evidence(FULL_READING)
    assert fused is not None
    assert set(detail.readings) == {"primary", "surface", "backup"}


def test_select_temperature_evidence_missing_primary_falls_back_to_two():
    reading = {k: v for k, v in FULL_READING.items() if k != "temp_primary"}
    fused, detail = select_temperature_evidence(reading)
    assert fused is not None
    assert set(detail.readings) == {"surface", "backup"}


def test_select_temperature_evidence_missing_backup_falls_back_to_two():
    reading = {k: v for k, v in FULL_READING.items() if k != "temp_backup"}
    fused, detail = select_temperature_evidence(reading)
    assert set(detail.readings) == {"primary", "surface"}


def test_select_temperature_evidence_missing_surface_falls_back_to_two():
    reading = {k: v for k, v in FULL_READING.items() if k != "temp_surface"}
    fused, detail = select_temperature_evidence(reading)
    assert set(detail.readings) == {"primary", "backup"}


def test_select_temperature_evidence_only_one_sensor_available():
    fused, detail = select_temperature_evidence({"temp_primary": 45.0})
    assert fused == pytest.approx(45.0)
    assert detail.reliability["primary"] == pytest.approx(P.SENSOR_PRIOR["reliable"])


def test_select_temperature_evidence_no_sensors_returns_none_not_a_fabricated_value():
    fused, detail = select_temperature_evidence({"voltage_dev": 0.05})
    assert fused is None
    assert detail is None


def test_select_temperature_evidence_ignores_non_finite_readings():
    reading = dict(FULL_READING)
    reading["temp_backup"] = float("nan")
    fused, detail = select_temperature_evidence(reading)
    assert set(detail.readings) == {"primary", "surface"}


def test_one_faulty_two_healthy_still_normalises_and_trusts_the_pair():
    reading = {"temp_primary": 120.0, "temp_surface": 44.0, "temp_backup": 45.0}
    fused, detail = select_temperature_evidence(reading)
    assert 40.0 <= fused <= 50.0
    assert sum(detail.subset_posterior.values()) == pytest.approx(1.0, abs=1e-9)
