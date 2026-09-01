"""Tests for sensor-reliability inference and Bayesian root-cause diagnosis."""

from __future__ import annotations

import numpy as np
import pytest

from config import model_parameters as P
from data.battery_simulator import simulate_pack
from models.fault_diagnosis import (
    diagnose_root_cause,
    expected_abs_voltage_dev,
    expected_log_gas,
    fuse_temperature,
    sensor_reliability,
)


def reading(**overrides) -> dict:
    """A calm, fully self-consistent reading of a healthy cell."""
    base = {
        "temp_primary": 33.0,
        "temp_backup": 33.2,
        "neighbour_temp": 32.4,
        "log_gas": expected_log_gas(33.0),
        "voltage_dev": expected_abs_voltage_dev(33.0),
        "current": 25.0,
        "soc": 0.62,
        "cooling_eff": 0.90,
        "temp_rate": 0.05,
        "temp_primary_prev": 32.9,
    }
    base.update(overrides)
    return base


def hot_but_genuine(temp: float = 92.0) -> dict:
    """A genuinely hot cell: every channel corroborates the primary sensor."""
    return reading(
        temp_primary=temp,
        temp_backup=temp - 0.4,
        neighbour_temp=temp - 4.0,
        log_gas=expected_log_gas(temp),
        voltage_dev=expected_abs_voltage_dev(temp),
        temp_rate=3.5,
        temp_primary_prev=temp - 0.6,
        cooling_eff=0.85,
    )


def hot_but_contradicted(temp: float = 92.0) -> dict:
    """THE demonstration case: hot primary, everything else says normal."""
    return reading(
        temp_primary=temp,
        temp_backup=33.0,
        neighbour_temp=32.0,
        log_gas=expected_log_gas(33.0),
        voltage_dev=expected_abs_voltage_dev(33.0),
        temp_rate=0.1,
        temp_primary_prev=temp - 0.3,
    )


# ---------------------------------------------------------------------------
# sensor reliability
# ---------------------------------------------------------------------------
def test_reliability_posterior_is_a_distribution():
    r = sensor_reliability(reading())
    assert r.p_reliable + r.p_faulty == pytest.approx(1.0)
    assert 0.0 <= r.p_faulty <= 1.0


def test_calm_consistent_reading_is_trusted():
    r = sensor_reliability(reading())
    assert r.p_faulty < 0.05
    assert r.fused_temp_c == pytest.approx(33.0, abs=0.5)


def test_conflicting_high_temperature_raises_sensor_fault_probability():
    """Hot primary + normal backup + no gas + normal voltage + cool neighbours.

    The system must prefer 'broken thermistor' over 'thermal runaway'.  This is
    derived from Bayes' rule, not from any hardcoded rule.
    """
    calm = sensor_reliability(reading())
    conflicted = sensor_reliability(hot_but_contradicted())

    assert conflicted.p_faulty > 0.95
    assert conflicted.p_faulty > calm.p_faulty
    # Having decided the primary is lying, the fused estimate must fall back
    # onto the backup channel.
    assert conflicted.fused_temp_c < 40.0


def test_genuinely_hot_cell_is_not_mistaken_for_a_sensor_fault():
    """The opposite failure mode: a real runaway must not be explained away."""
    r = sensor_reliability(hot_but_genuine())
    assert r.p_faulty < 0.30
    assert r.fused_temp_c > 80.0


def test_electrical_fault_does_not_look_like_a_sensor_fault():
    """Extra gas and extra voltage sag are evidence of a cell fault, not of a
    lying thermistor. This is why the gas/voltage cues are one-sided."""
    temp = 88.0
    r = sensor_reliability(
        reading(
            temp_primary=temp,
            temp_backup=temp - 0.3,
            neighbour_temp=48.0,             # single-cell event: neighbours lag
            log_gas=expected_log_gas(temp) + 1.4,     # MORE gas than expected
            voltage_dev=expected_abs_voltage_dev(temp) + 0.3,  # MORE sag
            temp_rate=4.0,
            temp_primary_prev=temp - 0.7,
        )
    )
    assert r.p_faulty < 0.30


def test_a_large_single_step_jump_is_suspicious():
    calm = sensor_reliability(reading())
    jumped = sensor_reliability(reading(temp_primary=33.0, temp_primary_prev=-12.0))
    assert jumped.p_faulty > calm.p_faulty


def test_reliability_works_without_a_previous_reading():
    r = sensor_reliability({k: v for k, v in reading().items() if k != "temp_primary_prev"})
    assert r.p_reliable + r.p_faulty == pytest.approx(1.0)
    assert len(r.cues) == 4


def test_fuse_temperature_interpolates_between_channels():
    assert fuse_temperature(90.0, 30.0, 0.0) == pytest.approx(90.0)
    assert fuse_temperature(90.0, 30.0, 1.0) == pytest.approx(30.0)
    assert fuse_temperature(90.0, 30.0, 0.5) == pytest.approx(60.0)


def test_expectation_curves_are_monotone_in_temperature():
    temps = np.linspace(30.0, 95.0, 40)
    gas = [expected_log_gas(t) for t in temps]
    vdev = [expected_abs_voltage_dev(t) for t in temps]
    assert all(b >= a for a, b in zip(gas, gas[1:]))
    assert all(b >= a for a, b in zip(vdev, vdev[1:]))


# ---------------------------------------------------------------------------
# root-cause diagnosis
# ---------------------------------------------------------------------------
def test_cause_posterior_is_a_distribution():
    r = diagnose_root_cause(reading())
    assert set(r.posterior) == set(P.CAUSES)
    assert sum(r.posterior.values()) == pytest.approx(1.0)
    assert all(0.0 <= v <= 1.0 for v in r.posterior.values())


def test_cause_posterior_stays_normalised_under_extreme_evidence():
    r = diagnose_root_cause(
        reading(temp_primary=500.0, voltage_dev=9.0, log_gas=40.0, current=1e4),
        p_sensor_faulty=1.0,
    )
    assert sum(r.posterior.values()) == pytest.approx(1.0)
    assert all(np.isfinite(v) for v in r.posterior.values())


def test_sensor_fault_evidence_selects_the_sensor_fault_cause():
    obs = hot_but_contradicted()
    sr = sensor_reliability(obs)
    r = diagnose_root_cause(obs, p_sensor_faulty=sr.p_faulty)
    assert r.top_cause == "Temperature Sensor Fault"


def test_cooling_collapse_selects_cooling_failure():
    temp = 68.0
    obs = reading(
        temp_primary=temp,
        temp_backup=temp - 0.2,
        neighbour_temp=temp - 2.0,          # whole pack is hot together
        log_gas=expected_log_gas(temp),
        voltage_dev=expected_abs_voltage_dev(temp),
        cooling_eff=0.16,                   # the giveaway
        current=42.0,
        temp_rate=0.8,
        temp_primary_prev=temp - 0.15,
    )
    r = diagnose_root_cause(obs, p_sensor_faulty=sensor_reliability(obs).p_faulty)
    assert r.top_cause == "Cooling-System Failure"


def test_localised_gassing_cell_selects_internal_short():
    temp = 95.0
    obs = reading(
        temp_primary=temp,
        temp_backup=temp - 0.3,
        neighbour_temp=50.0,                            # only this cell is hot
        log_gas=expected_log_gas(temp) + 1.3,           # gas beyond expectation
        voltage_dev=expected_abs_voltage_dev(temp) + 0.28,
        cooling_eff=0.88,
        current=20.0,
        temp_rate=2.6,
        temp_primary_prev=temp - 0.45,
    )
    r = diagnose_root_cause(obs, p_sensor_faulty=sensor_reliability(obs).p_faulty)
    assert r.top_cause == "Internal Short Circuit"


def test_soft_sensor_evidence_shifts_the_posterior_smoothly():
    """An uncertain reliability estimate should nudge, not dominate."""
    obs = hot_but_contradicted()
    p_low = diagnose_root_cause(obs, p_sensor_faulty=0.0).posterior["Temperature Sensor Fault"]
    p_mid = diagnose_root_cause(obs, p_sensor_faulty=0.5).posterior["Temperature Sensor Fault"]
    p_high = diagnose_root_cause(obs, p_sensor_faulty=1.0).posterior["Temperature Sensor Fault"]
    assert p_low < p_mid < p_high


def test_explanation_names_the_winning_cause():
    r = diagnose_root_cause(reading())
    assert r.top_cause in r.explanation
    assert r.top_probability == pytest.approx(max(r.posterior.values()))
    assert len(r.evidence_table) == len(P.CAUSE_FEATURES) + 1


# ---------------------------------------------------------------------------
# end-to-end against the simulator
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scenario", list(P.SCENARIOS))
def test_simulator_is_reproducible(scenario):
    a = simulate_pack(scenario, affected_cell=2, seed=7, n_steps=40)
    b = simulate_pack(scenario, affected_cell=2, seed=7, n_steps=40)
    assert a.frame.equals(b.frame)


def test_different_seeds_give_different_data():
    a = simulate_pack("Normal Operation", affected_cell=2, seed=1, n_steps=40)
    b = simulate_pack("Normal Operation", affected_cell=2, seed=2, n_steps=40)
    assert not a.frame["temp_primary"].equals(b.frame["temp_primary"])


def test_sensor_fault_scenario_keeps_the_pack_physically_healthy():
    """The injected fault must be in the sensor, not in the cell."""
    sim = simulate_pack("Temperature Sensor Fault", affected_cell=3, seed=11, n_steps=120)
    affected = sim.frame[sim.frame["cell"] == 3]
    assert (affected["true_state"] == P.S_HEALTHY).all()
    assert affected["temp_true"].max() < 45.0
    assert affected["temp_primary"].max() > 70.0        # the sensor lies
    assert affected["temp_backup"].max() < 45.0         # the backup does not


def test_internal_short_scenario_has_the_expected_signature():
    sim = simulate_pack("Internal Short Circuit", affected_cell=2, seed=11, n_steps=180)
    aff = sim.frame[sim.frame["cell"] == 2].sort_values("step").reset_index(drop=True)
    other = sim.frame[sim.frame["cell"] == 5].sort_values("step").reset_index(drop=True)

    assert aff["temp_true"].max() > 85.0                # runs away
    assert aff["temp_true"].max() > other["temp_true"].max() + 20.0   # localised
    assert aff["gas_ppm"].iloc[-1] > 50 * aff["gas_ppm"].iloc[0]      # gas rises
    assert aff["voltage_dev"].iloc[-1] > aff["voltage_dev"].iloc[0]   # voltage sags

    # Voltage must move before gas does: the short is electrical first.
    v_onset = int((aff["voltage_dev"] > aff["voltage_dev"].iloc[0] + 0.05).idxmax())
    g_onset = int((aff["gas_ppm"] > 5 * aff["gas_ppm"].iloc[0]).idxmax())
    assert v_onset < g_onset

    # Neighbours heat up, but with a delay.
    nb = sim.frame[sim.frame["cell"] == 1].sort_values("step").reset_index(drop=True)
    assert nb["temp_true"].iloc[-1] > nb["temp_true"].iloc[0] + 8.0


def test_external_heat_affects_the_whole_pack():
    sim = simulate_pack("External Heat Exposure", affected_cell=0, seed=5, n_steps=180)
    last = sim.frame[sim.frame["step"] == sim.n_steps - 1]
    assert (last["temp_true"] > 60.0).all()
    # Shallow gradient is the signature that distinguishes it from a short.
    assert last["temp_true"].max() - last["temp_true"].min() < 25.0


def test_simulator_rejects_bad_arguments():
    with pytest.raises(ValueError):
        simulate_pack("Not A Scenario")
    with pytest.raises(ValueError):
        simulate_pack("Normal Operation", affected_cell=99)
