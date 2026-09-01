"""
CellSafe-X : central parameter store.

EVERY prior, transition probability, Gaussian mean / standard deviation and
loss value used anywhere in the project is defined in this single module so
that the model can be inspected and explained during a viva.

IMPORTANT HONESTY NOTE
----------------------
These numbers are *prototype assumptions* chosen so that the probabilistic
machinery (Bayesian filtering, evidence fusion, expected-loss decisions) can be
demonstrated on synthetic data.  They are NOT certified physical parameters of
any real lithium-ion cell and were not fitted to real battery measurements.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# 1. TIME BASE
# ---------------------------------------------------------------------------
# One simulation step represents DT_SECONDS of wall-clock time.  All transition
# probabilities below are "per step", so the forecast horizons further down are
# converted into a number of steps using this constant.
DT_SECONDS: float = 10.0
DT_MINUTES: float = DT_SECONDS / 60.0

# Forecast horizons requested by the dashboard, in minutes.
FORECAST_HORIZONS_MIN = (5, 15, 30)


def horizon_steps(minutes: float) -> int:
    """Convert a horizon in minutes into an integer number of HMM steps."""
    return max(1, int(round(minutes / DT_MINUTES)))


# ---------------------------------------------------------------------------
# 2. HIDDEN STATE SPACE
# ---------------------------------------------------------------------------
# The latent variable Z_t of the Hidden Markov Model.  Index order is fixed and
# used consistently by every module (filter, forecast, decision engine).
STATES = ("Healthy", "Abnormal Heating", "Pre-Runaway", "Thermal Runaway")
N_STATES = len(STATES)

S_HEALTHY, S_ABNORMAL, S_PRE, S_RUNAWAY = 0, 1, 2, 3

# States that count as "dangerous" for risk forecasting and for the evaluation
# of missed events / false alarms.
DANGEROUS_STATES = (S_PRE, S_RUNAWAY)

# Colours used by the dashboard for each hidden state (green -> red ramp).
STATE_COLORS = {
    "Healthy": "#1f9d55",           # green
    "Abnormal Heating": "#d69e2e",  # amber
    "Pre-Runaway": "#dd6b20",       # orange
    "Thermal Runaway": "#c53030",   # red
}

# ---------------------------------------------------------------------------
# 3. INITIAL STATE DISTRIBUTION  P(Z_0)
# ---------------------------------------------------------------------------
# Prior belief before any sensor reading is seen.  A pack is *usually* healthy
# when the vehicle is switched on, but we keep non-zero mass on the other states
# so that the filter can react quickly and never divides by zero.
INITIAL_STATE_PROBS = np.array([0.90, 0.07, 0.025, 0.005], dtype=float)

# ---------------------------------------------------------------------------
# 4. STATE TRANSITION MATRIX  P(Z_t | Z_{t-1})
# ---------------------------------------------------------------------------
# Row = state at t-1, column = state at t.  Encoded behaviour:
#   * a cell mostly stays where it is (strong diagonal),
#   * deterioration is gradual: Healthy -> Abnormal -> Pre-Runaway -> Runaway,
#   * limited recovery is allowed from Abnormal Heating (cooling catches up) and,
#     more rarely, from Pre-Runaway,
#   * Healthy cannot jump straight to Pre-Runaway or Runaway - a cell must pass
#     through Abnormal Heating first (no "teleporting" failures),
#   * Thermal Runaway is *nearly* absorbing: 0.999 self-loop.
#
# The Healthy -> Abnormal rate is deliberately small (0.004 per 10 s step, about
# 2.4% per minute).  A larger value would make a perfectly healthy pack forecast
# an implausibly high 30-minute danger probability purely from the prior.
TRANSITION_MATRIX = np.array(
    [
        # ->Healthy  ->Abnormal  ->Pre     ->Runaway
        [0.9960,     0.0040,     0.0000,   0.0000],   # from Healthy
        [0.0300,     0.9600,     0.0100,   0.0000],   # from Abnormal Heating
        [0.0000,     0.0150,     0.9650,   0.0200],   # from Pre-Runaway
        [0.0000,     0.0000,     0.0010,   0.9990],   # from Thermal Runaway
    ],
    dtype=float,
)

# ---------------------------------------------------------------------------
# 5. OBSERVATION MODEL  P(Y_t | Z_t)
# ---------------------------------------------------------------------------
# Continuous sensor channels modelled with a state-conditional Gaussian.
# Channel meanings:
#   temp_c        trusted (sensor-fusion) cell temperature        [degC]
#   temp_rate     rate of change of the trusted temperature       [degC/min]
#   voltage_dev   |deviation of terminal voltage from nominal|    [V]
#   log_gas       log1p of vent-gas concentration                 [log ppm]
#   neighbour_c   temperature of the hottest neighbouring cell    [degC]
#
# WHICH SENSORS ARE *NOT* IN THE EMISSION MODEL, AND WHY
# ------------------------------------------------------
# `current`, `soc` and `cooling_eff` are all measured and all displayed, but
# none of them is an *emission* of the cell's hidden thermal state.  They are
# inputs to the system, not symptoms of it: a coolant pump can fail while the
# cell it serves is still perfectly cold, and a pack can be charged hard while
# every cell stays healthy.  Putting cooling_eff into P(Y|Z) would make the
# filter declare "Pre-Runaway" the moment the pump degrades, before any cell has
# actually heated up - which is exactly the kind of spurious alarm this project
# is trying to avoid.  These three channels are therefore routed to the
# root-cause classifier, which asks *why* a cell is hot rather than *how hot*
# it is.
OBS_CHANNELS = ("temp_c", "temp_rate", "voltage_dev", "log_gas", "neighbour_c")

# means[state][channel]
OBS_MEANS = {
    "Healthy":          {"temp_c": 32.0, "temp_rate": 0.05, "voltage_dev": 0.010, "log_gas": 1.10, "neighbour_c": 31.0},
    "Abnormal Heating": {"temp_c": 48.0, "temp_rate": 0.50, "voltage_dev": 0.060, "log_gas": 2.56, "neighbour_c": 38.0},
    "Pre-Runaway":      {"temp_c": 68.0, "temp_rate": 1.80, "voltage_dev": 0.180, "log_gas": 4.39, "neighbour_c": 50.0},
    "Thermal Runaway":  {"temp_c": 95.0, "temp_rate": 4.50, "voltage_dev": 0.400, "log_gas": 5.86, "neighbour_c": 68.0},
}

# Standard deviations are intentionally wide so that adjacent hidden states
# overlap substantially.  Narrow, well-separated Gaussians would make the
# evaluation look artificially perfect, which would be dishonest for a
# synthetic-data prototype.
OBS_STDS = {
    "Healthy":          {"temp_c": 6.0,  "temp_rate": 0.35, "voltage_dev": 0.030, "log_gas": 0.80, "neighbour_c": 5.0},
    "Abnormal Heating": {"temp_c": 8.0,  "temp_rate": 0.60, "voltage_dev": 0.050, "log_gas": 1.00, "neighbour_c": 7.0},
    "Pre-Runaway":      {"temp_c": 9.0,  "temp_rate": 1.00, "voltage_dev": 0.090, "log_gas": 1.10, "neighbour_c": 9.0},
    "Thermal Runaway":  {"temp_c": 12.0, "temp_rate": 1.80, "voltage_dev": 0.150, "log_gas": 1.00, "neighbour_c": 11.0},
}

# The five channels are treated as conditionally independent given the hidden
# state (a naive-Bayes emission model).  They are in reality correlated - gas
# and voltage deviation are both driven by temperature - so the raw product of
# five densities would be over-confident.  We therefore temper each
# log-likelihood by a weight < 1, which is equivalent to saying "these five
# correlated channels together carry about 2.6 independent channels' worth of
# evidence".  This is a documented approximation, not a physical law.
#
# `neighbour_c` gets the smallest weight: a hot neighbour is only weak evidence
# about *this* cell, and a single cell in runaway is genuinely much hotter than
# the cells around it.
OBS_LIKELIHOOD_WEIGHTS = {
    "temp_c": 0.70,
    "temp_rate": 0.60,
    "voltage_dev": 0.50,
    "log_gas": 0.50,
    "neighbour_c": 0.35,
}

# ---------------------------------------------------------------------------
# 6. SENSOR RELIABILITY MODEL
# ---------------------------------------------------------------------------
# Binary hypothesis H in {reliable, faulty} for the PRIMARY temperature sensor
# of a given cell.  A temperature sensor failing is uncommon but far from
# impossible over the life of a pack.
SENSOR_PRIOR = {"reliable": 0.97, "faulty": 0.03}

# Each reliability cue is a residual that should be ~0 when the primary sensor
# agrees with the rest of the evidence.  Under H=reliable the residual is tight;
# under H=faulty the primary reading is decoupled from physical reality, so the
# residual is modelled as a broad zero-mean Gaussian (large discrepancies become
# plausible, small ones are not impossible - the sensor may be drifting slowly).
#
# Cues:
#   backup_delta     T_primary - T_backup.  The single strongest cue: two
#                    independent thermistors on the same cell should agree.
#   neighbour_delta  T_primary - T_neighbour.  A DELIBERATELY WEAK cue (large
#                    sd_r): a cell in genuine runaway really is far hotter than
#                    its neighbours, so disagreement here is only mild evidence
#                    of a sensor problem.
#   gas_shortfall    min(0, observed log-gas - log-gas expected at T_primary)
#   volt_shortfall   min(0, observed |voltage dev| - value expected at T_primary)
#   jump_residual    T_primary(t) - T_primary(t-1); a real thermal mass cannot
#                    change by tens of degrees in one 10-second step.
#
# WHY THE GAS AND VOLTAGE CUES ARE ONE-SIDED
# ------------------------------------------
# Only a SHORTFALL is evidence against the primary sensor.  If the sensor claims
# 90 degC but there is no vent gas and no voltage deviation, the claimed
# temperature is not corroborated and the sensor is suspect.  The opposite -
# MORE gas or MORE voltage deviation than the temperature explains - is evidence
# of an electrical fault in the cell, not of a broken thermistor.  Using a
# two-sided residual here caused the module to declare a sensor fault during a
# genuine internal short circuit, which is exactly the wrong answer.
SENSOR_CUE_MODEL = {
    #                    mean_reliable  std_reliable  mean_faulty  std_faulty
    "backup_delta":    {"mu_r": 0.0, "sd_r": 1.5,  "mu_f": 0.0, "sd_f": 20.0},
    "neighbour_delta": {"mu_r": 3.0, "sd_r": 12.0, "mu_f": 3.0, "sd_f": 30.0},
    "gas_shortfall":   {"mu_r": 0.0, "sd_r": 1.2,  "mu_f": 0.0, "sd_f": 3.2},
    "volt_shortfall":  {"mu_r": 0.0, "sd_r": 0.05, "mu_f": 0.0, "sd_f": 0.22},
    "jump_residual":   {"mu_r": 0.0, "sd_r": 1.2,  "mu_f": 0.0, "sd_f": 9.0},
}

# Expected physical signature at a given temperature, used to build the
# residuals above.  Piecewise-linear interpolation knots (temperature -> value).
# "If the cell really were this hot, we would expect roughly this much gas and
#  this much voltage deviation."
EXPECTED_LOG_GAS_KNOTS = ([30.0, 45.0, 60.0, 75.0, 95.0], [1.10, 2.20, 3.60, 4.80, 5.90])
EXPECTED_ABS_VDEV_KNOTS = ([30.0, 45.0, 60.0, 75.0, 95.0], [0.010, 0.050, 0.120, 0.230, 0.400])

# Temperature used as the "trusted" fallback when the primary sensor is judged
# unreliable: the posterior-weighted blend of primary and backup readings.
# (see models/fault_diagnosis.py :: fuse_temperature)

# ---------------------------------------------------------------------------
# 7. ROOT-CAUSE MODEL
# ---------------------------------------------------------------------------
CAUSES = (
    "Normal/Fast-Charging Heat",
    "Temperature Sensor Fault",
    "Cooling-System Failure",
    "Overcharging",
    "Internal Short Circuit",
    "External Heat Exposure",
)

# P(Cause) before looking at any evidence.  Benign heating is the most common
# explanation for a warm cell; catastrophic causes are rarer.
CAUSE_PRIORS = {
    "Normal/Fast-Charging Heat": 0.34,
    "Temperature Sensor Fault": 0.12,
    "Cooling-System Failure": 0.15,
    "Overcharging": 0.13,
    "Internal Short Circuit": 0.11,
    "External Heat Exposure": 0.15,
}

# Ambient reference temperature used to express "temperature excess" features.
AMBIENT_C = 30.0

# Feature vector used by the root-cause classifier.  Unlike the HMM, this model
# deliberately works on the RAW primary temperature, so that a sensor fault
# remains visible as an implausibly hot reading rather than being silently
# corrected away before it can be diagnosed.
#
# Two of the features are *anomalies* rather than raw readings:
#
#   vdev_anomaly = |voltage deviation| - the deviation expected at this temperature
#   gas_anomaly  = log-gas             - the log-gas expected at this temperature
#
# This matters.  Raw gas and raw voltage deviation are almost pure functions of
# temperature (hot electrolyte decomposes and vents, hot cells drift), so as raw
# features they carry no information about *why* the cell is hot - every cause
# at 85 degC looks alike.  The anomaly, on the other hand, is exactly the part
# the temperature does not explain, and that residual is highly diagnostic:
# an internal short vents and sags more than its temperature warrants, while a
# lying thermistor produces neither the gas nor the voltage drift that its
# claimed temperature demands.
#
# `self_vs_neighbour` (T_primary - T_neighbour) separates a localised fault
# (one cell far hotter than the rest of the pack) from a pack-wide one (all six
# cells heating together).
CAUSE_FEATURES = (
    "temp_excess",        # T_primary - ambient                                [degC]
    "temp_rate",          # dT/dt of the raw primary channel                   [degC/min]
    "vdev_anomaly",       # |voltage dev| - expected at T_primary              [V]
    "gas_anomaly",        # log-gas - expected at T_primary                    [log ppm]
    "current",            # pack current, positive = charging                  [A]
    "soc",                # state of charge                                    [0-1]
    "cooling_eff",        # cooling loop effectiveness                         [0-1]
    "self_vs_neighbour",  # T_primary - hottest neighbour                      [degC]
)

CAUSE_MEANS = {
    "Normal/Fast-Charging Heat": {"temp_excess": 12.0, "temp_rate": 0.40, "vdev_anomaly":  0.000, "gas_anomaly":  0.00, "current": 60.0, "soc": 0.70, "cooling_eff": 0.86, "self_vs_neighbour":  1.0},
    "Temperature Sensor Fault":  {"temp_excess": 45.0, "temp_rate": 1.50, "vdev_anomaly": -0.200, "gas_anomaly": -3.00, "current": 25.0, "soc": 0.60, "cooling_eff": 0.88, "self_vs_neighbour": 45.0},
    "Cooling-System Failure":    {"temp_excess": 25.0, "temp_rate": 0.70, "vdev_anomaly":  0.000, "gas_anomaly":  0.00, "current": 40.0, "soc": 0.60, "cooling_eff": 0.22, "self_vs_neighbour":  2.0},
    "Overcharging":              {"temp_excess": 18.0, "temp_rate": 0.60, "vdev_anomaly":  0.160, "gas_anomaly":  0.30, "current": 95.0, "soc": 0.96, "cooling_eff": 0.82, "self_vs_neighbour":  2.0},
    "Internal Short Circuit":    {"temp_excess": 38.0, "temp_rate": 2.20, "vdev_anomaly":  0.280, "gas_anomaly":  1.20, "current": 20.0, "soc": 0.55, "cooling_eff": 0.86, "self_vs_neighbour": 30.0},
    "External Heat Exposure":    {"temp_excess": 32.0, "temp_rate": 1.00, "vdev_anomaly":  0.000, "gas_anomaly":  0.00, "current":  8.0, "soc": 0.50, "cooling_eff": 0.60, "self_vs_neighbour":  0.0},
}

# Wide standard deviations: several causes genuinely look alike early on, and we
# do not want the classifier to be falsely certain.
CAUSE_STDS = {
    "Normal/Fast-Charging Heat": {"temp_excess": 10.0, "temp_rate": 0.50, "vdev_anomaly": 0.040, "gas_anomaly": 0.70, "current": 30.0, "soc": 0.20, "cooling_eff": 0.08, "self_vs_neighbour":  3.0},
    "Temperature Sensor Fault":  {"temp_excess": 20.0, "temp_rate": 2.00, "vdev_anomaly": 0.120, "gas_anomaly": 1.50, "current": 35.0, "soc": 0.25, "cooling_eff": 0.08, "self_vs_neighbour": 25.0},
    "Cooling-System Failure":    {"temp_excess": 12.0, "temp_rate": 0.60, "vdev_anomaly": 0.040, "gas_anomaly": 0.70, "current": 35.0, "soc": 0.25, "cooling_eff": 0.13, "self_vs_neighbour":  4.0},
    "Overcharging":              {"temp_excess": 10.0, "temp_rate": 0.55, "vdev_anomaly": 0.080, "gas_anomaly": 0.80, "current": 28.0, "soc": 0.06, "cooling_eff": 0.10, "self_vs_neighbour":  4.0},
    "Internal Short Circuit":    {"temp_excess": 16.0, "temp_rate": 1.40, "vdev_anomaly": 0.120, "gas_anomaly": 0.90, "current": 32.0, "soc": 0.25, "cooling_eff": 0.10, "self_vs_neighbour": 18.0},
    "External Heat Exposure":    {"temp_excess": 14.0, "temp_rate": 0.90, "vdev_anomaly": 0.040, "gas_anomaly": 0.70, "current": 25.0, "soc": 0.28, "cooling_eff": 0.12, "self_vs_neighbour":  4.0},
}

# Same tempering idea as the HMM emission model: eight correlated features would
# otherwise produce a degenerate 100% / 0% posterior.
CAUSE_LIKELIHOOD_WEIGHTS = {
    "temp_excess": 0.45,
    "temp_rate": 0.45,
    "vdev_anomaly": 0.55,
    "gas_anomaly": 0.55,
    "current": 0.40,
    "soc": 0.45,
    "cooling_eff": 0.55,
    "self_vs_neighbour": 0.50,
}

# Soft evidence link between the sensor-reliability module and the root-cause
# module:  P(primary sensor is faulty | Cause).  Only the sensor-fault cause
# expects a faulty sensor; everything else expects a working one.
P_SENSOR_FAULTY_GIVEN_CAUSE = {
    "Normal/Fast-Charging Heat": 0.04,
    "Temperature Sensor Fault": 0.90,
    "Cooling-System Failure": 0.05,
    "Overcharging": 0.04,
    "Internal Short Circuit": 0.03,
    "External Heat Exposure": 0.05,
}

# ---------------------------------------------------------------------------
# 8. DECISION MODEL
# ---------------------------------------------------------------------------
ACTIONS = (
    "Continue Monitoring",
    "Request Backup Measurement",
    "Reduce Charging Current",
    "Increase Cooling",
    "Isolate Affected Module",
    "Emergency Shutdown",
)

# Loss (cost) matrix  L(action, hidden state), on a 0-100 scale where
# 100 = worst acceptable outcome.  Each entry is the sum of:
#   * safety loss              - harm if the state really is dangerous and we
#                                under-react,
#   * battery damage           - stress the action itself puts on the pack,
#   * vehicle unavailability   - how much driving capability we give up,
#   * false-shutdown cost      - penalty for stopping a perfectly healthy pack,
#   * measurement delay        - cost of spending a step gathering more data.
#
# Reading the matrix:  doing nothing is cheap when Healthy and catastrophic in
# Thermal Runaway; Emergency Shutdown is the reverse.  The optimum therefore
# shifts smoothly as the posterior mass moves right.
LOSS_MATRIX = np.array(
    [
        # Healthy  Abnormal  Pre-Runaway  Thermal Runaway
        [  0.0,     18.0,      70.0,        100.0],  # Continue Monitoring
        [  3.0,      9.0,      45.0,         88.0],  # Request Backup Measurement
        [  9.0,      7.0,      30.0,         72.0],  # Reduce Charging Current
        [ 11.0,      6.0,      26.0,         66.0],  # Increase Cooling
        [ 38.0,     28.0,      14.0,         34.0],  # Isolate Affected Module
        [ 62.0,     50.0,      22.0,          8.0],  # Emergency Shutdown
    ],
    dtype=float,
)

# Breakdown of the same matrix, purely for display in the dashboard so a
# reviewer can see what the numbers are meant to represent.  Values are the
# dominant cost component of each action.
ACTION_COST_NOTES = {
    "Continue Monitoring": "No intervention cost; entire loss comes from safety risk if the state is actually dangerous.",
    "Request Backup Measurement": "Small measurement-delay cost; buys evidence before committing to an intervention.",
    "Reduce Charging Current": "Mild vehicle-performance cost; removes electrical heat input.",
    "Increase Cooling": "Mild energy cost; removes thermal energy without limiting the vehicle.",
    "Isolate Affected Module": "High availability cost (reduced range) but contains a single bad cell.",
    "Emergency Shutdown": "Very high false-shutdown / unavailability cost, minimal safety loss.",
}

# Optional "probabilistic intervention assessment": a coarse model of how much
# each action is expected to reduce the one-step dangerous-state probability.
# This is a modelling assumption about intervention effectiveness - it is NOT
# formal causal counterfactual inference and must not be described as such.
INTERVENTION_EFFECTIVENESS = {
    "Continue Monitoring": 0.00,
    "Request Backup Measurement": 0.02,
    "Reduce Charging Current": 0.25,
    "Increase Cooling": 0.35,
    "Isolate Affected Module": 0.70,
    "Emergency Shutdown": 0.85,
}

# ---------------------------------------------------------------------------
# 9. THRESHOLD BASELINE
# ---------------------------------------------------------------------------
# The naive system we compare against: a single fixed temperature trip point on
# the primary sensor, exactly like a classical BMS over-temperature cut-out.
THRESHOLD_WARN_C = 55.0
THRESHOLD_CRITICAL_C = 70.0

# ---------------------------------------------------------------------------
# 10. PACK GEOMETRY / SIMULATOR DEFAULTS
# ---------------------------------------------------------------------------
N_CELLS = 6
DEFAULT_SEED = 42
DEFAULT_STEPS = 180  # 180 steps x 10 s = 30 simulated minutes
NOMINAL_VOLTAGE = 3.70

SCENARIOS = (
    "Normal Operation",
    "Fast Charging",
    "Temperature Sensor Fault",
    "Cooling-System Failure",
    "Internal Short Circuit",
    "External Heat Exposure",
)

# Which root cause is the *ground truth* for each simulated scenario.  Used by
# the evaluation script to score the root-cause classifier.
SCENARIO_TRUE_CAUSE = {
    "Normal Operation": "Normal/Fast-Charging Heat",
    "Fast Charging": "Normal/Fast-Charging Heat",
    "Temperature Sensor Fault": "Temperature Sensor Fault",
    "Cooling-System Failure": "Cooling-System Failure",
    "Internal Short Circuit": "Internal Short Circuit",
    "External Heat Exposure": "External Heat Exposure",
}

# ---------------------------------------------------------------------------
# 11. SANITY CHECKS (run at import time - cheap and catches typos immediately)
# ---------------------------------------------------------------------------
def _validate() -> None:
    assert INITIAL_STATE_PROBS.shape == (N_STATES,)
    assert abs(INITIAL_STATE_PROBS.sum() - 1.0) < 1e-9, "initial distribution must sum to 1"
    assert TRANSITION_MATRIX.shape == (N_STATES, N_STATES)
    row_sums = TRANSITION_MATRIX.sum(axis=1)
    assert np.allclose(row_sums, 1.0), f"transition rows must sum to 1, got {row_sums}"
    assert (TRANSITION_MATRIX >= 0).all(), "transition probabilities must be non-negative"
    assert abs(sum(CAUSE_PRIORS.values()) - 1.0) < 1e-9, "cause priors must sum to 1"
    assert abs(sum(SENSOR_PRIOR.values()) - 1.0) < 1e-9, "sensor prior must sum to 1"
    assert LOSS_MATRIX.shape == (len(ACTIONS), N_STATES)
    for s in STATES:
        assert set(OBS_MEANS[s]) == set(OBS_CHANNELS)
        assert set(OBS_STDS[s]) == set(OBS_CHANNELS)
        assert all(v > 0 for v in OBS_STDS[s].values())
    for c in CAUSES:
        assert set(CAUSE_MEANS[c]) == set(CAUSE_FEATURES)
        assert set(CAUSE_STDS[c]) == set(CAUSE_FEATURES)
        assert all(v > 0 for v in CAUSE_STDS[c].values())


_validate()
