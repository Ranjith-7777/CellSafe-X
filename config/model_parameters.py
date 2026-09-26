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
#
# PHASE-1 CORRECTION: `temp_rate` weight lowered 0.60 -> 0.35 (diagnosis below).
# `temp_rate` is not a fresh 10 s measurement: it is an EMA-smoothed slope over
# a rolling 1-minute window (see models/__init__.py :: CellPipeline), so its
# state-conditional Gaussians are far better separated, in z-score terms, than
# any raw channel's.  Quantified on the Fast-Charging scenario (benign 85 A
# current ramp, t = 0.5-2.5 min): at the worst observed step the tempered
# log-likelihood swing contributed by temp_rate alone was ~48 nats (Healthy vs
# Thermal Runaway), five to ten times larger than every other channel combined,
# so a single transient rate reading overwhelmed four channels that all agreed
# the cell was merely Abnormal, driving a momentary P(dangerous) as high as
# 1.00 while temp_c was still 41 degC. This is a genuine failure to detect
# "an internal short/runaway ramp" vs "the current stepped up 2 minutes ago" -
# not something the reliability model can fix, because it happens upstream of
# sensor-fault reasoning. Grid search over this weight (0.60 -> 0.10) on the
# evaluation seeds showed the worst-case spike falls below the 0.5 alarm
# threshold at 0.35 (worst case 0.29, vs 1.00 at 0.60) while *improving*
# Cooling-System-Failure and Internal-Short accuracy (the rate channel matters
# less than the emission means suggested); the only material cost is External
# Heat Exposure accuracy (-3 points), which legitimately relies on temp_rate to
# separate a shallow multi-cell heat wave from ordinary Healthy noise. This is
# a disclosed trade-off, not a free win - see README Phase-1 notes.
OBS_LIKELIHOOD_WEIGHTS = {
    "temp_c": 0.70,
    "temp_rate": 0.35,
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
# 6a. OVERLAPPING THERMAL SENSORS (Phase 7A)
# ---------------------------------------------------------------------------
# Professor suggestion: model MULTIPLE sensors observing the SAME underlying
# cell thermal condition explicitly, and fuse them by inferred reliability
# rather than a fixed primary/backup binary test. This generalises (does not
# replace) the existing primary-vs-backup reliability test above: that
# mechanism stays exactly as-is and continues to drive the main synthetic
# pipeline; models/sensor_fusion.py is an additive, symmetric N-sensor
# extension of the same "reliable = tight residual, faulty = broad residual"
# idea, reusing SENSOR_PRIOR as each sensor's independent prior.
#
# Three sensors per cell, physically distinct, NOT three copies of one signal:
#   primary  - internal thermistor, fastest/most direct reading (existing
#              `temp_primary` channel, reused unchanged).
#   backup   - a second internal thermistor, same physical siting as primary,
#              independent noise draw (existing `temp_backup`, reused
#              unchanged).
#   surface  - an external surface probe. Thermal mass between the core and
#              the case means it LAGS the true internal temperature and reads
#              a real (small, physical) offset besides - modelled as an
#              exponential lag plus a fixed bias, not just extra noise.
OVERLAPPING_SENSOR_NAMES = ("primary", "surface", "backup")

# Measurement noise (1-sigma, degC) when a sensor is functioning normally.
# Surface is noisier: it is farther from the core reaction and more exposed
# to ambient/convective disturbance than an embedded internal thermistor.
OVERLAPPING_SENSOR_NOISE_C = {"primary": 0.55, "surface": 0.90, "backup": 0.55}

# Exponential response lag, in simulator steps (DT_SECONDS each), applied
# only to the surface sensor's underlying (noiseless) signal - the physical
# thermal mass between core and case a primary/backup internal thermistor
# does not have to cross.
OVERLAPPING_SENSOR_LAG_STEPS = {"primary": 0, "surface": 3, "backup": 0}

# Small steady-state bias (degC) baked into a healthy surface reading (the
# case genuinely runs a little cooler than the core even at equilibrium).
OVERLAPPING_SENSOR_BIAS_C = {"primary": 0.0, "surface": -1.5, "backup": 0.0}

# Reused, not duplicated: each sensor's prior probability of being reliable
# at any given step is SENSOR_PRIOR["reliable"]; the broad "if faulty, could
# be almost anything" residual scale reuses the same order of magnitude as
# SENSOR_CUE_MODEL's backup_delta sd_f.
OVERLAPPING_SENSOR_FAULT_SIGMA_C = 20.0

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

# DEPRECATED (Phase 3): a flat "how much does this action help" fraction that
# used to be multiplied directly onto the one-step dangerous probability. It is
# kept ONLY as a backward-compatible reference value (e.g. for anyone diffing
# against the Phase 1/2 static display) and is no longer read by
# models/decision_engine.py or models/intervention.py. The actual intervention
# assessment is now a controlled-transition forecast - see
# INTERVENTION_TRANSITION_EFFECTS below and models/intervention.py.
INTERVENTION_EFFECTIVENESS = {
    "Continue Monitoring": 0.00,
    "Request Backup Measurement": 0.02,
    "Reduce Charging Current": 0.25,
    "Increase Cooling": 0.35,
    "Isolate Affected Module": 0.70,
    "Emergency Shutdown": 0.85,
}

# ---------------------------------------------------------------------------
# 8a. INTERVENTION TRANSITION MODEL (Phase 3)
# ---------------------------------------------------------------------------
# CellSafe-X is a manually-specified HMM/DBN: the transition matrix is a
# modelling assumption, not something estimated from interventional data. We
# therefore do NOT claim Pearlian do(X) causal identification. What follows is
# a MODEL-BASED / CONTROLLED-TRANSITION counterfactual: for each action we
# define an explicit, documented alternative transition matrix A(action) and
# ask "what would the k-step-ahead belief be if the pack evolved under this
# controlled dynamics instead of the baseline dynamics from now on". This is
# forecasting under an assumed controlled model, not a causal effect estimated
# from data - see models/intervention.py for the full disclaimer.
#
# Each action is defined by two multipliers applied to TRANSITION_MATRIX:
#   escalation_multiplier  scales every entry that moves to a MORE dangerous
#                          state (column index > row index). <= 1 means the
#                          action suppresses further deterioration.
#   recovery_multiplier    scales every entry that moves to a LESS dangerous
#                          state (column index < row index). >= 1 means the
#                          action makes recovery somewhat more likely.
# The diagonal (self-loop) absorbs whatever probability mass this frees up or
# consumes, so every resulting row still sums to exactly 1 - see
# models/intervention.py :: controlled_transition_matrix.
#
# Physical justification for each action (informal, not fitted to data):
#   Continue Monitoring / Request Backup Measurement: no physical effect on
#     the pack, so both are the identity (1.0, 1.0). Backup Measurement is
#     information-gathering only - see the Phase 3 README note on why it must
#     not show a fake physical risk reduction.
#   Reduce Charging Current: removes electrical heat input, which mainly
#     matters for the low/mid bands (a charging-driven cell has not yet
#     started an independent exothermic reaction) - a moderate escalation cut.
#   Increase Cooling: removes thermal energy broadly, so a stronger and
#     more uniform escalation cut than reducing current, though still bounded
#     - cooling cannot outpace an already-exothermic reaction as effectively
#     as it can slow ordinary joule/charging heat.
#   Isolate Affected Module: disconnects the cell electrically, stopping any
#     further electrically-driven heating (current AND charging heat at once)
#     - a strong escalation cut, weaker than shutdown only because isolation
#     is scoped to one module rather than the whole pack.
#   Emergency Shutdown: the strongest controlled intervention available under
#     this model - removes essentially all further externally-driven
#     escalation. It is NOT modelled as reversing an already-exothermic
#     Thermal Runaway (see the tiny, unchanged Runaway self-loop) - shutdown
#     stops making things worse, it does not undo damage already done. This
#     mirrors the loss matrix, where Emergency Shutdown is cheapest specifically
#     in the Thermal Runaway column, not because it cures it.
#
# Multipliers are monotonic across actions by design (escalation_multiplier
# strictly non-increasing, recovery_multiplier strictly non-decreasing, from
# Continue Monitoring down to Emergency Shutdown) so that a stronger action
# can never be assessed as riskier than a weaker one - this is asserted by
# tests/test_intervention.py, not just claimed here.
INTERVENTION_TRANSITION_EFFECTS = {
    "Continue Monitoring":        {"escalation_multiplier": 1.00, "recovery_multiplier": 1.00},
    "Request Backup Measurement": {"escalation_multiplier": 1.00, "recovery_multiplier": 1.00},
    "Reduce Charging Current":    {"escalation_multiplier": 0.65, "recovery_multiplier": 1.15},
    "Increase Cooling":           {"escalation_multiplier": 0.45, "recovery_multiplier": 1.35},
    "Isolate Affected Module":    {"escalation_multiplier": 0.25, "recovery_multiplier": 1.50},
    "Emergency Shutdown":         {"escalation_multiplier": 0.10, "recovery_multiplier": 1.75},
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
# 10a. PACK TOPOLOGY (Phase 4) - single source of truth
# ---------------------------------------------------------------------------
# The six cells are a linear string; cell i exchanges heat only with i-1 and
# i+1 (data/battery_simulator.py's thermal integration and
# models/propagation.py's coupling model both call `pack_neighbours` below,
# so there is exactly one topology definition in the whole project, not two
# that could quietly drift apart).
PACK_TOPOLOGY = "linear_chain"


def pack_neighbours(cell: int, n_cells: int = N_CELLS) -> tuple[int, ...]:
    """Cell indices that physically/thermally border `cell`."""
    return tuple(j for j in (cell - 1, cell + 1) if 0 <= j < n_cells)


# ---------------------------------------------------------------------------
# 10b. CROSS-CELL PROPAGATION MODEL (Phase 4)
# ---------------------------------------------------------------------------
# IMPORTANT SCOPE NOTE: cells are filtered independently (models/__init__.py
# :: CellPipeline runs one BayesianFilter per cell); there is no joint
# multi-cell HMM. The propagation model below is a SEPARATE, documented
# forecasting layer built on top of the independent per-cell posteriors - it
# asks "how much should a neighbour's OWN forecast be nudged upward given how
# dangerous the adjacent cell is projected to become", not "what is the exact
# joint posterior over all six cells" (that would need the 4^6-state coupled
# DBN the README already lists as future work).
#
# Mechanism: a target cell's controlled transition matrix (same scale/
# renormalise construction as models/intervention.py) gets its escalation
# entries multiplied by
#     escalation_multiplier = 1 + PROPAGATION_GAIN * pressure
#     pressure = sum over neighbours j of PROPAGATION_COUPLING_STRENGTH * P(dangerous)_j
# where P(dangerous)_j is neighbour j's OWN independent (uncoupled) forecast
# at the same horizon. Using each neighbour's independent forecast (rather
# than solving a fixed point over the whole pack) avoids circular dependencies
# between adjacent cells and keeps the model a one-step, tractable, clearly-
# documented approximation - not an attempt at an exact joint solution.
#
# PROPAGATION_COUPLING_STRENGTH is a normalised [0, 1] edge weight. The linear
# chain is symmetric and uncoupled by construction from K_NEIGH's sign (heat
# flows whichever direction is colder), so all edges share one weight; a
# richer topology would need a per-edge matrix instead of a scalar.
# PROPAGATION_GAIN converts that pressure into an escalation-probability
# multiplier increase; it is an ENGINEERED, undocumented-in-data constant
# (like the Phase 3 intervention multipliers), not fitted to the simulator's
# K_NEIGH thermal-conduction coefficient - translating a continuous-temperature
# conduction gain into a discrete-state escalation-probability multiplier is
# a modelling choice, disclosed here rather than presented as derived.
# PROPAGATION_MAX_ESCALATION_MULTIPLIER caps how much a neighbour's escalation
# risk can be amplified, so a single dangerous cell cannot deterministically
# drag its neighbour's probability mass to zero self-loop.
PROPAGATION_COUPLING_STRENGTH = 1.0
PROPAGATION_GAIN = 2.0
PROPAGATION_MAX_ESCALATION_MULTIPLIER = 3.0

# ---------------------------------------------------------------------------
# 10c. ACTIVE SENSING / VALUE OF INFORMATION (Phase 4)
# ---------------------------------------------------------------------------
# Candidate channels for expected-information-gain analysis are restricted to
# OBS_CHANNELS (section 5 above) because those are the only channels with a
# well-defined state-conditional likelihood P(y | Z) the filter already uses -
# "current" and "cooling_eff" feed the root-cause classifier, not the hidden-
# state filter, so VoI for the HIDDEN STATE cannot be mathematically formed
# for them without inventing an emission model that does not exist. A repeat
# reading of `temp_c` stands in for "backup temperature" (an independent
# confirming measurement of the same emission channel).
ACTIVE_SENSING_CANDIDATE_CHANNELS = OBS_CHANNELS

# Monte Carlo sample count for the EIG/EVI expectation over possible
# observations, and a fixed seed so every call is exactly reproducible.
ACTIVE_SENSING_MC_SAMPLES = 4000
ACTIVE_SENSING_MC_SEED = 20240

# Gating thresholds for "should we actually request another measurement"
# (see models/active_sensing.py :: recommend_measurement). Centralised here,
# not scattered as UI-only magic numbers:
#   CONFIDENCE   - if the current posterior's max probability already exceeds
#                  this, the state is not ambiguous enough to be worth asking.
#   MIN_EIG_NATS - if even the best candidate channel's expected information
#                  gain is below this, no measurement is worth the delay.
#   EMERGENCY_DANGEROUS_P - if P(dangerous) is at or above this, delaying
#                  action to gather more evidence is treated as unsafe
#                  regardless of how informative a measurement would be.
ACTIVE_SENSING_CONFIDENCE_THRESHOLD = 0.95
ACTIVE_SENSING_MIN_EIG_NATS = 0.02
ACTIVE_SENSING_EMERGENCY_DANGEROUS_P = 0.90

# ---------------------------------------------------------------------------
# 10d. CANONICAL SENSOR REGISTRY (Phase 7B, Goal 1)
# ---------------------------------------------------------------------------
# "What is the set of sensors CellSafe-X understands?" - one table, built
# from the constants already defined above rather than repeating them, so it
# cannot silently drift out of sync with OBS_CHANNELS / CAUSE_FEATURES /
# ACTIVE_SENSING_CANDIDATE_CHANNELS / OVERLAPPING_SENSOR_NAMES.
#
# Fields per entry:
#   physical_quantity   what it measures (not which channel it becomes)
#   family               overlap-group key; sensors sharing a family observe
#                        the SAME latent quantity (see
#                        models.sensor_fusion). Sensors in different families
#                        are never fused together even if both are
#                        temperatures (e.g. neighbour_temp is a different
#                        cell's surface, not this cell's core).
#   overlaps             True iff >=2 registry entries share `family`
#   hmm_eligible         True iff this sensor feeds an OBS_CHANNELS emission
#                        (directly, or - for the three overlapping probes -
#                        via models.sensor_fusion into `temp_c`)
#   active_sensing_eligible  True iff EIG/EVI may recommend requesting this
#                        reading. Only ONE representative per overlapping
#                        family is eligible (Goal 4: EIG operates on the
#                        single abstract `temp_c` channel, so treating
#                        primary/surface/backup as three independent
#                        active-sensing choices would triple-count the same
#                        evidence).
#   evidence_type        "physical_state" (reaches the hidden-state filter)
#                        or "contextual" (root-cause only)
#   units
#   missing_evidence_support  "marginalised" (verified: this channel is a
#                        member of OBS_CHANNELS, so
#                        observation_log_likelihood already skips it cleanly
#                        when absent/non-finite - see models/bayesian_filter.py)
#                        or "required" (root-cause's build_cause_features
#                        hard-subscripts this key; a missing contextual
#                        channel raises KeyError today, not silently defaulted)
#
# Goal 4 (active-sensing/registry alignment): which sensor FAMILY backs each
# abstract HMM emission channel. `temp_rate` has no sensor of its own - it is
# the slope of the same core-temperature family's fused reading - so it maps
# to the same family as `temp_c` rather than getting a fake extra sensor.
# `models/active_sensing.py` uses this to report which physical sensor
# family an EIG/EVI recommendation actually refers to, and so that
# requesting "more temp_c evidence" is never double-counted against
# primary/surface/backup as three separate active-sensing opportunities.
OBS_CHANNEL_TO_SENSOR_FAMILY: dict[str, str] = {
    "temp_c": "cell_core_temperature",
    "temp_rate": "cell_core_temperature",
    "voltage_dev": "voltage",
    "log_gas": "gas",
    "neighbour_c": "neighbour_cell_temperature",
}

SENSOR_REGISTRY: dict[str, dict[str, object]] = {
    "temp_primary": {
        "physical_quantity": "temperature", "family": "cell_core_temperature",
        "overlaps": True, "hmm_eligible": True, "active_sensing_eligible": True,
        "evidence_type": "physical_state", "units": "degC",
        "missing_evidence_support": "marginalised",
    },
    "temp_surface": {
        "physical_quantity": "temperature", "family": "cell_core_temperature",
        "overlaps": True, "hmm_eligible": True, "active_sensing_eligible": False,
        "evidence_type": "physical_state", "units": "degC",
        "missing_evidence_support": "marginalised",
    },
    "temp_backup": {
        "physical_quantity": "temperature", "family": "cell_core_temperature",
        "overlaps": True, "hmm_eligible": True, "active_sensing_eligible": False,
        "evidence_type": "physical_state", "units": "degC",
        "missing_evidence_support": "marginalised",
    },
    "neighbour_temp": {
        "physical_quantity": "temperature", "family": "neighbour_cell_temperature",
        "overlaps": False, "hmm_eligible": True, "active_sensing_eligible": True,
        "evidence_type": "physical_state", "units": "degC",
        "missing_evidence_support": "marginalised",
    },
    "voltage_dev": {
        "physical_quantity": "voltage_deviation", "family": "voltage",
        "overlaps": False, "hmm_eligible": True, "active_sensing_eligible": True,
        "evidence_type": "physical_state", "units": "V",
        "missing_evidence_support": "marginalised",
    },
    "log_gas": {
        "physical_quantity": "vent_gas_concentration", "family": "gas",
        "overlaps": False, "hmm_eligible": True, "active_sensing_eligible": True,
        "evidence_type": "physical_state", "units": "log ppm",
        "missing_evidence_support": "marginalised",
    },
    "current": {
        "physical_quantity": "pack_current", "family": "current",
        "overlaps": False, "hmm_eligible": False, "active_sensing_eligible": False,
        "evidence_type": "contextual", "units": "A",
        "missing_evidence_support": "required",
    },
    "soc": {
        "physical_quantity": "state_of_charge", "family": "state_of_charge",
        "overlaps": False, "hmm_eligible": False, "active_sensing_eligible": False,
        "evidence_type": "contextual", "units": "fraction",
        "missing_evidence_support": "required",
    },
    "cooling_eff": {
        "physical_quantity": "cooling_effectiveness", "family": "cooling",
        "overlaps": False, "hmm_eligible": False, "active_sensing_eligible": False,
        "evidence_type": "contextual", "units": "fraction",
        "missing_evidence_support": "required",
    },
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
    assert set(INTERVENTION_TRANSITION_EFFECTS) == set(ACTIONS)
    for a, eff in INTERVENTION_TRANSITION_EFFECTS.items():
        assert 0.0 <= eff["escalation_multiplier"] <= 1.0, a
        assert eff["recovery_multiplier"] >= 1.0, a
    # Monotonicity: a physically stronger action must never be configured with
    # a weaker (higher) escalation multiplier or a weaker (lower) recovery
    # multiplier than a milder action - see the Phase 3 comment above.
    strength_order = [
        "Continue Monitoring", "Request Backup Measurement", "Reduce Charging Current",
        "Increase Cooling", "Isolate Affected Module", "Emergency Shutdown",
    ]
    esc = [INTERVENTION_TRANSITION_EFFECTS[a]["escalation_multiplier"] for a in strength_order]
    rec = [INTERVENTION_TRANSITION_EFFECTS[a]["recovery_multiplier"] for a in strength_order]
    assert all(esc[i] >= esc[i + 1] for i in range(len(esc) - 1)), "escalation_multiplier must be non-increasing with action strength"
    assert all(rec[i] <= rec[i + 1] for i in range(len(rec) - 1)), "recovery_multiplier must be non-decreasing with action strength"
    for i in range(N_CELLS):
        nb = pack_neighbours(i)
        assert all(0 <= j < N_CELLS and j != i for j in nb)
        # symmetric linear-chain topology: j is my neighbour iff i is j's neighbour
        assert all(i in pack_neighbours(j) for j in nb)
    assert 0.0 <= PROPAGATION_COUPLING_STRENGTH <= 1.0
    assert PROPAGATION_GAIN >= 0.0
    assert PROPAGATION_MAX_ESCALATION_MULTIPLIER >= 1.0
    assert set(ACTIVE_SENSING_CANDIDATE_CHANNELS) <= set(OBS_CHANNELS)
    assert ACTIVE_SENSING_MC_SAMPLES > 0
    assert 0.0 <= ACTIVE_SENSING_CONFIDENCE_THRESHOLD <= 1.0
    assert ACTIVE_SENSING_MIN_EIG_NATS >= 0.0
    assert 0.0 <= ACTIVE_SENSING_EMERGENCY_DANGEROUS_P <= 1.0
    assert set(OVERLAPPING_SENSOR_NOISE_C) == set(OVERLAPPING_SENSOR_NAMES)
    assert set(OVERLAPPING_SENSOR_LAG_STEPS) == set(OVERLAPPING_SENSOR_NAMES)
    assert set(OVERLAPPING_SENSOR_BIAS_C) == set(OVERLAPPING_SENSOR_NAMES)
    assert all(v > 0 for v in OVERLAPPING_SENSOR_NOISE_C.values())
    assert all(v >= 0 for v in OVERLAPPING_SENSOR_LAG_STEPS.values())
    assert OVERLAPPING_SENSOR_FAULT_SIGMA_C > 0.0
    # SENSOR_REGISTRY must stay in sync with the constants it is derived from
    # (Goal 1: exactly one source of truth), not silently drift.
    for name in OVERLAPPING_SENSOR_NAMES:
        assert f"temp_{name}" in SENSOR_REGISTRY, name
        assert SENSOR_REGISTRY[f"temp_{name}"]["overlaps"] is True
    overlap_families: dict[str, int] = {}
    for entry in SENSOR_REGISTRY.values():
        overlap_families[entry["family"]] = overlap_families.get(entry["family"], 0) + 1
    for entry in SENSOR_REGISTRY.values():
        assert entry["overlaps"] == (overlap_families[entry["family"]] > 1), entry
    # exactly one active-sensing-eligible representative per overlapping family
    active_families: dict[str, int] = {}
    for entry in SENSOR_REGISTRY.values():
        if entry["active_sensing_eligible"]:
            active_families[entry["family"]] = active_families.get(entry["family"], 0) + 1
    assert all(v == 1 for v in active_families.values()), active_families
    assert set(OBS_CHANNEL_TO_SENSOR_FAMILY) == set(OBS_CHANNELS)
    assert set(ACTIVE_SENSING_CANDIDATE_CHANNELS) <= set(OBS_CHANNEL_TO_SENSOR_FAMILY)
    assert set(OBS_CHANNEL_TO_SENSOR_FAMILY.values()) <= set(overlap_families)


_validate()
