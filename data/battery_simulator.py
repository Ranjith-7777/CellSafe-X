"""
Six-cell synthetic battery-pack simulator.

EVERYTHING PRODUCED BY THIS MODULE IS SIMULATED.  No real cell, sensor or
vehicle data is used anywhere in CellSafe-X.

The pack is a linear string of six cells; cell i exchanges heat with cells
i-1 and i+1.  Each cell is a lumped thermal mass integrated with a forward
Euler step:

    T[t+1] = T[t] + dt * ( q_gen(t) - k * cooling_eff(t) * (T[t] - T_amb(t))
                            + k_n * (T_neigh_mean - T[t]) )  + process noise

The ground-truth hidden state label is derived from the *true* temperature and
the *true* heating rate (both invisible to the inference pipeline), so the
evaluation is scoring the filter's ability to recover a latent variable from
noisy, partially contradictory observations - not scoring it against its own
output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from config import model_parameters as P
from models.fault_diagnosis import expected_abs_voltage_dev, expected_log_gas

# --- thermal constants (prototype assumptions, not certified cell physics) ---
K_COOL = 0.35          # cooling gain per minute at cooling_eff = 1
K_NEIGH = 0.10         # cell-to-cell conduction gain per minute
PROC_NOISE_C = 0.10    # process noise on the true temperature, degC per step

# --- sensor noise (prototype assumptions) -----------------------------------
SENSOR_NOISE_C = 0.55        # primary / backup thermistor noise, degC
NEIGH_SENSOR_NOISE_C = 0.60
GAS_LOG_NOISE = 0.45         # multiplicative (log-space) noise on gas
VDEV_NOISE = 0.012
COOLING_NOISE = 0.025
CURRENT_NOISE = 3.0

# --- ground-truth labelling thresholds on the TRUE temperature --------------
LABEL_ABNORMAL_C = 42.0
LABEL_PRE_C = 58.0
LABEL_RUNAWAY_C = 82.0
LABEL_RATE_ESCALATE = 2.5    # degC/min of true heating that escalates one band

# PHASE-2 CORRECTION: `true_rate` is a single-step (10 s) finite difference of
# the TRUE temperature, computed AFTER that step's process noise
# (PROC_NOISE_C = 0.10 degC/step) is added - so it carries roughly
# 0.10 / dt_min =~ 0.6 degC/min of 1-sigma noise on top of the real physics.
# A one-off noisy sample crossing LABEL_RATE_ESCALATE therefore does not mean
# the cell is truly escalating. Phase-2 audit of the evaluation's 566 "missed
# dangerous event" samples found: 288 ground-truth dangerous-state segments in
# the evaluation set, median length 1 step (10 s); 199 of 288 segments were
# <=3 steps and NONE of those ever reached the actual Pre-Runaway temperature
# (LABEL_PRE_C); 405 one-step A-B-A state reversals were found across the
# dataset. This is single-step label flicker driven by process noise on the
# rate channel, not a real physical event - the same failure mode Phase 1
# fixed in the model's own emission likelihood (see OBS_LIKELIHOOD_WEIGHTS
# comment in config/model_parameters.py), except here it was baked into the
# ground truth itself.
#
# Fix: require the elevated-rate condition to hold for RATE_ESCALATE_MIN_STEPS
# consecutive steps (20 s) before it escalates the label a band. This does NOT
# change any physical threshold (LABEL_ABNORMAL_C / LABEL_PRE_C / LABEL_RUNAWAY_C
# are untouched) and does not make any scenario's ground truth "easier" - a
# sustained genuine heating-rate excursion (as seen in Internal Short Circuit
# and the late stages of Cooling-System Failure) is unaffected; only isolated,
# single-step noise crossings are no longer promoted to a dangerous label.
RATE_ESCALATE_MIN_STEPS = 2


@dataclass
class SimulationResult:
    """Container for one simulated run."""

    frame: pd.DataFrame                     # long format, one row per (step, cell)
    scenario: str
    affected_cell: int
    seed: int
    n_steps: int
    dt_seconds: float = P.DT_SECONDS
    notes: str = ""
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def n_cells(self) -> int:
        return int(self.frame["cell"].nunique())


def _label_state(true_temp: float, true_rate: float, rate_escalate_streak: int = 0) -> int:
    """Ground-truth hidden state from the simulator's internal physical state.

    `rate_escalate_streak` is the number of CONSECUTIVE prior steps (including
    this one) for which `true_rate >= LABEL_RATE_ESCALATE` has held, tracked by
    the caller. Escalation only fires once that streak reaches
    `RATE_ESCALATE_MIN_STEPS`, so a single noisy rate sample cannot flip the
    label (see the RATE_ESCALATE_MIN_STEPS comment above).
    """
    if true_temp >= LABEL_RUNAWAY_C:
        return P.S_RUNAWAY
    if true_temp >= LABEL_PRE_C:
        return P.S_PRE
    if true_temp >= LABEL_ABNORMAL_C:
        base = P.S_ABNORMAL
    else:
        base = P.S_HEALTHY
    # Rapid, SUSTAINED heating is itself a symptom: escalate one band (never
    # past Pre). `true_rate` alone is not used here - see RATE_ESCALATE_MIN_STEPS.
    if rate_escalate_streak >= RATE_ESCALATE_MIN_STEPS and base < P.S_PRE:
        base += 1
    return base


def _ramp(t: float, start: float, duration: float) -> float:
    """Smooth 0 -> 1 ramp starting at `start` minutes, over `duration` minutes."""
    if duration <= 0:
        return 1.0 if t >= start else 0.0
    x = (t - start) / duration
    return float(np.clip(x, 0.0, 1.0))


def simulate_pack(
    scenario: str = "Normal Operation",
    affected_cell: int = 2,
    seed: int = P.DEFAULT_SEED,
    n_steps: int = P.DEFAULT_STEPS,
    n_cells: int = P.N_CELLS,
) -> SimulationResult:
    """Generate a reproducible synthetic time series for the whole pack.

    Parameters
    ----------
    scenario       one of config.model_parameters.SCENARIOS
    affected_cell  index (0-based) of the cell that carries the fault
    seed           RNG seed; the same seed always reproduces the same run
    n_steps        number of DT_SECONDS steps to simulate
    """
    if scenario not in P.SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; expected one of {P.SCENARIOS}")
    if not (0 <= affected_cell < n_cells):
        raise ValueError(f"affected_cell must be in [0, {n_cells - 1}]")

    rng = np.random.default_rng(seed)
    dt = P.DT_MINUTES

    T = np.full(n_cells, P.AMBIENT_C + 2.0, dtype=float) + rng.normal(0, 0.4, n_cells)
    T_prev = T.copy()
    soc = np.full(n_cells, 0.62, dtype=float)

    rows: List[Dict[str, object]] = []
    sensor_offset = np.zeros(n_cells, dtype=float)   # additive primary-sensor fault
    gas_mult = np.ones(n_cells, dtype=float)
    extra_vdev = np.zeros(n_cells, dtype=float)
    rate_escalate_streak = np.zeros(n_cells, dtype=int)
    # Phase 7A: a third overlapping thermal sensor (surface probe). It lags
    # the true core temperature through an exponential filter representing
    # the thermal mass between core and case - see
    # config.model_parameters.OVERLAPPING_SENSOR_LAG_STEPS.
    T_surface_lag = T.copy()
    surface_alpha = 1.0 / (P.OVERLAPPING_SENSOR_LAG_STEPS["surface"] + 1)
    surface_lag_history: List[np.ndarray] = []

    for step in range(n_steps):
        t_min = step * dt

        # ------------------------------------------------------------------
        # scenario-specific drivers for this step
        # ------------------------------------------------------------------
        amb = np.full(n_cells, P.AMBIENT_C, dtype=float)
        cooling = np.full(n_cells, 0.90, dtype=float)
        q_gen = np.full(n_cells, 0.95, dtype=float)   # baseline self-heating
        current = 25.0
        d_soc = -0.0006
        sensor_offset[:] = 0.0
        gas_mult[:] = 1.0
        extra_vdev[:] = 0.0

        if scenario == "Normal Operation":
            pass

        elif scenario == "Fast Charging":
            r = _ramp(t_min, 0.5, 2.0)
            current = 25.0 + 60.0 * r
            q_gen[:] = 0.95 + 5.2 * r
            cooling[:] = 0.86
            # SOC deliberately stops well short of full: a pack charged to 0.99
            # at high current genuinely *is* closer to overcharging, and we want
            # this scenario to represent benign fast-charge heat instead.
            d_soc = 0.0014
            extra_vdev[:] = 0.015 * r

        elif scenario == "Temperature Sensor Fault":
            # Physics is identical to Normal Operation; only the primary
            # thermistor of the affected cell misbehaves.
            r = _ramp(t_min, 6.0, 0.4)
            sensor_offset[affected_cell] = 55.0 * r

        elif scenario == "Cooling-System Failure":
            r = _ramp(t_min, 2.0, 14.0)
            cooling[:] = 0.90 - 0.78 * r
            q_gen[:] = 2.4
            q_gen[affected_cell] = 2.9          # hottest position in the pack
            current = 45.0
            d_soc = 0.0010

        elif scenario == "Internal Short Circuit":
            r = _ramp(t_min, 5.0, 7.0)
            q_gen[affected_cell] = 0.95 + 26.0 * r
            # Voltage sags first, gas appears only once the cell is hot enough
            # for the electrolyte to decompose - a deliberate delay.
            extra_vdev[affected_cell] = 0.34 * _ramp(t_min, 4.5, 4.0)
            gas_mult[affected_cell] = 1.0 + 3.5 * _ramp(t_min, 8.5, 6.0)
            current = 20.0
            d_soc = -0.0008

        elif scenario == "External Heat Exposure":
            r = _ramp(t_min, 3.0, 12.0)
            # An external source heats the pack from one side.  The gradient is
            # shallow on purpose: the defining signature of external heating is
            # that the WHOLE pack warms together, unlike an internal short where
            # a single cell runs away far ahead of its neighbours.
            for i in range(n_cells):
                dist = abs(i - affected_cell)
                amb[i] = P.AMBIENT_C + 58.0 * r * (1.0 - 0.07 * dist)
            cooling[:] = 0.60
            q_gen[:] = 0.95
            current = 5.0
            d_soc = -0.0003

        # ------------------------------------------------------------------
        # thermal integration (true, hidden physics)
        # ------------------------------------------------------------------
        neigh_mean = np.zeros(n_cells, dtype=float)
        for i in range(n_cells):
            nb = P.pack_neighbours(i, n_cells)
            neigh_mean[i] = float(np.mean([T[j] for j in nb]))

        T_prev = T.copy()
        dT = dt * (
            q_gen
            - K_COOL * cooling * (T - amb)
            + K_NEIGH * (neigh_mean - T)
        )
        T = T + dT + rng.normal(0.0, PROC_NOISE_C, n_cells)
        T = np.clip(T, P.AMBIENT_C - 5.0, 160.0)
        true_rate = (T - T_prev) / dt
        T_surface_lag = T_surface_lag + (T - T_surface_lag) * surface_alpha

        rate_escalate_streak = np.where(
            true_rate >= LABEL_RATE_ESCALATE, rate_escalate_streak + 1, 0
        )

        soc = np.clip(soc + d_soc, 0.02, 1.0)

        # ------------------------------------------------------------------
        # sensing (what the inference pipeline is allowed to see)
        # ------------------------------------------------------------------
        for i in range(n_cells):
            nb = P.pack_neighbours(i, n_cells)
            neigh_true = float(np.max([T[j] for j in nb]))

            temp_primary = T[i] + sensor_offset[i] + rng.normal(0.0, SENSOR_NOISE_C)
            temp_backup = T[i] + rng.normal(0.0, SENSOR_NOISE_C)
            neigh_meas = neigh_true + rng.normal(0.0, NEIGH_SENSOR_NOISE_C)

            # Gas follows the TRUE temperature (electrolyte decomposition), with
            # a scenario multiplier for the internal short.
            base_log_gas = expected_log_gas(float(T[i]))
            log_gas = base_log_gas + float(np.log(gas_mult[i])) + rng.normal(0.0, GAS_LOG_NOISE)
            log_gas = float(max(0.0, log_gas))
            gas_ppm = float(np.expm1(log_gas))

            base_vdev = expected_abs_voltage_dev(float(T[i]))
            # This field is a deviation magnitude.  Noise can otherwise make a
            # near-zero deviation slightly negative, which is physically
            # meaningless and inconsistent with every downstream consumer.
            voltage_dev = abs(base_vdev + extra_vdev[i] + rng.normal(0.0, VDEV_NOISE))

            cooling_meas = float(np.clip(cooling[i] + rng.normal(0.0, COOLING_NOISE), 0.0, 1.0))
            current_meas = float(current + rng.normal(0.0, CURRENT_NOISE))

            state = _label_state(float(T[i]), float(true_rate[i]), int(rate_escalate_streak[i]))

            rows.append(
                {
                    "step": step,
                    "time_min": round(t_min, 4),
                    "cell": i,
                    "scenario": scenario,
                    "is_affected": bool(i == affected_cell),
                    # ground truth (hidden from the pipeline)
                    "temp_true": float(T[i]),
                    "true_rate": float(true_rate[i]),
                    "true_state": int(state),
                    "true_state_name": P.STATES[state],
                    "sensor_faulty_true": bool(abs(sensor_offset[i]) > 5.0),
                    # observations (visible to the pipeline)
                    "temp_primary": float(temp_primary),
                    "temp_backup": float(temp_backup),
                    "temp_surface": float(T_surface_lag[i]),  # patched below with its own noise
                    "neighbour_temp": float(neigh_meas),
                    "gas_ppm": gas_ppm,
                    "log_gas": log_gas,
                    "voltage_dev": float(voltage_dev),
                    "voltage_v": float(P.NOMINAL_VOLTAGE - voltage_dev),
                    "current": current_meas,
                    "soc": float(soc[i]),
                    "cooling_eff": cooling_meas,
                }
            )
        surface_lag_history.append(T_surface_lag.copy())

    # Third overlapping sensor (Phase 7A), drawn in ONE batch AFTER the
    # entire step loop so the RNG stream consumed by every pre-existing
    # channel (across every step) is byte-for-byte unchanged - reproducible
    # frozen-baseline behaviour is fully preserved.
    surface_lag_arr = np.concatenate(surface_lag_history)
    surface_noise = rng.normal(0.0, P.OVERLAPPING_SENSOR_NOISE_C["surface"], surface_lag_arr.shape)
    temp_surface_arr = surface_lag_arr + P.OVERLAPPING_SENSOR_BIAS_C["surface"] + surface_noise
    for row, val in zip(rows, temp_surface_arr):
        row["temp_surface"] = float(val)

    frame = pd.DataFrame(rows)

    notes = {
        "Normal Operation": "Steady moderate discharge. All six cells stay near ambient; nothing should trip.",
        "Fast Charging": "High charge current warms the whole pack uniformly into the Abnormal Heating band. This is benign heat, not a fault.",
        "Temperature Sensor Fault": "The physics is identical to Normal Operation. Only the primary thermistor on the affected cell reports a large false offset from t = 6 min.",
        "Cooling-System Failure": "Cooling effectiveness collapses from 0.90 to ~0.12 over 14 minutes. The whole pack heats, the affected cell leads.",
        "Internal Short Circuit": "The affected cell develops an internal short at t = 5 min: voltage sags first, temperature rises steeply, vent gas appears with a delay, and neighbours heat later by conduction.",
        "External Heat Exposure": "An external heat source raises the effective ambient on one side of the pack. Multiple cells heat together with no gas and no voltage anomaly.",
    }[scenario]

    return SimulationResult(
        frame=frame,
        scenario=scenario,
        affected_cell=affected_cell,
        seed=seed,
        n_steps=n_steps,
        notes=notes,
        metadata={
            "data_source": "SIMULATED - synthetic data generated by data/battery_simulator.py",
            "true_cause": P.SCENARIO_TRUE_CAUSE[scenario],
            "n_cells": n_cells,
        },
    )
